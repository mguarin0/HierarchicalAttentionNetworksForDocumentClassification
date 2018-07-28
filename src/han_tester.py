"""
@author: Michael Guarino
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import pickle
from scipy import stats
from collections import Counter
import os
from han import HAN
from utils import prjPaths, get_logger
from dataProcessing import IMDB

def get_flags():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  tf.flags.DEFINE_string("dataset", "imdb",
                         "enter the type of training dataset")
  tf.flags.DEFINE_string("run_type", "val",
                         "enter val or test to specify run_type (default: val)")
  tf.flags.DEFINE_integer("log_summaries_every", 30,
                          "Save model summaries after this many steps (default: 30)")
  tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.90,
                        "gpu memory to be used (default: 0.90)")
  tf.flags.DEFINE_boolean("wait_for_checkpoint_files", False,
                        "wait for model checkpoint file to be created")

  FLAGS = tf.flags.FLAGS
  FLAGS._parse_flags()

  return FLAGS
# end

def get_most_recently_created_file(files):
  return max(files, key=os.path.getctime) # most recently created file in list of files
# end

if __name__ == '__main__':

  MINUTE = 60
  paths = prjPaths()
  FLAGS = get_flags()

  print("current version of tf:{}".format(tf.__version__))

  assert(FLAGS.run_type == "val" or FLAGS.run_type == "test")

  print("loading persisted variables...")
  with open(os.path.join(paths.LIB_DIR, FLAGS.dataset, "persisted_vars.p"), "rb") as handle:
    persisted_vars = pickle.load(handle)

  # create new graph set as default
  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  gpu_options=gpu_options)
    session_conf.gpu_options.allocator_type = "BFC"

    # create new session set it as default
    with tf.Session(config=session_conf) as sess:

      # create han model instance
      han = HAN(max_seq_len=persisted_vars["max_seq_len"],
                max_sent_len=persisted_vars["max_sent_len"],
                num_classes=persisted_vars["num_classes"],
                vocab_size=persisted_vars["vocab_size"],
                embedding_size=persisted_vars["embedding_dim"],
                max_grad_norm=persisted_vars["max_grad_norm"],
                dropout_keep_proba=persisted_vars["dropout_keep_proba"],
                learning_rate=persisted_vars["learning_rate"])

      global_step = tf.Variable(0, name="global_step", trainable=False)
      tvars = tf.trainable_variables()
      grads, global_norm = tf.clip_by_global_norm(tf.gradients(han.loss, tvars),
                                                  han.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(han.learning_rate)
      test_op = optimizer.apply_gradients(zip(grads, tvars),
                                           name="{}_op".format(FLAGS.run_type),
                                           global_step=global_step)

      # write summaries 
      merge_summary_op = tf.summary.merge_all()
      test_summary_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, FLAGS.run_type), sess.graph)

      # give check for checkpoint files directory if none then sleep until a checkpoint is created
      #if os.listdir(paths.CHECKPOINT_DIR) == []:
        #time.sleep(2*MINUTE)

      meta_file = get_most_recently_created_file([os.path.join(paths.CHECKPOINT_DIR, file) for file in os.listdir(paths.CHECKPOINT_DIR) if file.endswith('.meta')])
      saver = tf.train.import_meta_graph(meta_file)

      # Initialize all variables
      sess.run(tf.global_variables_initializer())

      def test_step(sample_num, x_batch, y_batch, docsize, sent_size, is_training):

        feed_dict = {han.input_x: x_batch,
                     han.input_y: y_batch,
                     han.sentence_lengths: docsize,
                     han.word_lengths: sent_size,
                     han.is_training: is_training}

        loss, accuracy = sess.run([han.loss, han.accuracy], feed_dict=feed_dict)
        return loss, accuracy
      # end

      # generate batches on imdb dataset else quit
      if FLAGS.dataset == "imdb":
        dataset_controller = IMDB(action="fetch")
      else:
        exit("set dataset flag to appropiate dataset")

      x, y, docsize, sent_size = dataset_controller.get_data(type_=FLAGS.run_type) # fetch dataset
      all_evaluated_chkpts = [] # list of all checkpoint files previously evaluated

      # testing loop
      while True:

        if FLAGS.wait_for_checkpoint_files:
          time.sleep(2*MINUTE) # wait to allow for creation of new checkpoint file
        else:
          time.sleep(0*MINUTE) # don't wait for model checkpoint files

        # if checkpoint file already evaluated then continue and wait for a new checkpoint file
        if (tf.train.latest_checkpoint(paths.CHECKPOINT_DIR) in all_evaluated_chkpts):
          continue

        # restore most recent checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(paths.CHECKPOINT_DIR)) # restore most recent checkpoint
        all_evaluated_chkpts.append(tf.train.latest_checkpoint(paths.CHECKPOINT_DIR)) # add current checkpoint to list of evaluated checkpoints

        losses = [] # aggregate testing losses on a given checkpoint
        accuracies = [] # aggregate testing accuracies on a given checkpoint

        tic = time.time() # start time for step

        # loop to test every sample on a given checkpoint
        for i, batch in enumerate(tqdm(list(zip(x, y, docsize, sent_size)))):

          x_batch, y_batch, docsize_batch, sent_size_batch = batch
          x_batch = np.expand_dims(x_batch, axis=0)
          y_batch = np.expand_dims(y_batch, axis=0)
          sent_size_batch = np.expand_dims(sent_size_batch, axis=0)

          # run step
          loss, accuracy = test_step(sample_num=i,
                                    x_batch=x_batch,
                                    y_batch=y_batch,
                                    docsize=docsize,
                                    sent_size=sent_size,
                                    is_training=False)
          losses.append(loss)
          accuracies.append(accuracy)

        time_elapsed = time.time() - tic # end time for step

        losses_accuracies_vars = {"losses": losses, "accuracies": accuracies}

        print("Time taken to complete {} evaluation of {} checkpoint: {}".format(FLAGS.run_type, all_evaluated_chkpts[-1], time_elapsed))
        for k in losses_accuracies_vars.keys():
          print("stats for {}: {}".format(k, stats.describe(losses_accuracies_vars[k])))
          print(Counter(losses_accuracies_vars[k]))

        filename, ext = os.path.splitext(all_evaluated_chkpts[-1])
        pickle._dump(losses_accuracies_vars, open(os.path.join(paths.LIB_DIR, FLAGS.dataset, "losses_accuracies_vars_{}.p".format(filename.split("/")[-1])), "wb"))

    sess.close()
