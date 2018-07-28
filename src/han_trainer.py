"""
@author: Michael Guarino
"""

import tensorflow as tf
import numpy as np
import time
import pickle
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
  tf.flags.DEFINE_string("run_type", "train",
                         "enter train or test to specify run_type (default: train)")
  tf.flags.DEFINE_integer("embedding_dim", 100,
                          "Dimensionality of character embedding (default: 100)")
  tf.flags.DEFINE_integer("batch_size", 2,
                          "Batch Size (default: 2)")
  tf.flags.DEFINE_integer("num_epochs", 25,
                          "Number of training epochs (default: 25)")
  tf.flags.DEFINE_integer("evaluate_every", 100,
                          "Evaluate model on dev set after this many steps")
  tf.flags.DEFINE_integer("log_summaries_every", 30,
                          "Save model summaries after this many steps (default: 30)")
  tf.flags.DEFINE_integer("checkpoint_every", 100,
                          "Save model after this many steps (default: 100)")
  tf.flags.DEFINE_integer("num_checkpoints", 5,
                          "Number of checkpoints to store (default: 5)")
  tf.flags.DEFINE_float("max_grad_norm", 5.0,
                        "maximum permissible norm of the gradient (default: 5.0)")
  tf.flags.DEFINE_float("dropout_keep_proba", 0.5,
                        "probability of neurons turned off (default: 0.5)")
  tf.flags.DEFINE_float("learning_rate", 0.001,
                        "model learning rate (default: 0.001)")
  tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.90,
                        "gpu memory to be used (default: 0.90)")

  FLAGS = tf.flags.FLAGS
  FLAGS._parse_flags()

  return FLAGS
# end

if __name__ == '__main__':

  paths = prjPaths()
  FLAGS = get_flags()

  print("current version of tf:{}".format(tf.__version__))

  assert(FLAGS.run_type == "train")

  print("loading persisted variables...")

  with open(os.path.join(paths.LIB_DIR, FLAGS.dataset, "persisted_vars.p"), "rb") as handle:
    persisted_vars = pickle.load(handle)

  persisted_vars["embedding_dim"] = FLAGS.embedding_dim
  persisted_vars["max_grad_norm"] = FLAGS.max_grad_norm
  persisted_vars["dropout_keep_proba"] = FLAGS.dropout_keep_proba
  persisted_vars["learning_rate"] = FLAGS.learning_rate
  pickle._dump(persisted_vars, open(os.path.join(paths.LIB_DIR, FLAGS.dataset, "persisted_vars.p"), "wb"))

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
      train_op = optimizer.apply_gradients(zip(grads, tvars),
                                           name="train_op",
                                           global_step=global_step)

      # write summaries 
      merge_summary_op = tf.summary.merge_all()
      train_summary_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, FLAGS.run_type), sess.graph)

      # checkpoint model
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

      # Initialize all variables
      sess.run(tf.global_variables_initializer())

      def train_step(epoch, x_batch, y_batch, docsize, sent_size, is_training):
        tic = time.time() # start time for step

        feed_dict = {han.input_x: x_batch,
                     han.input_y: y_batch,
                     han.sentence_lengths: docsize,
                     han.word_lengths: sent_size,
                     han.is_training: is_training}

        _, step, loss, accuracy, summaries = sess.run([train_op, global_step, han.loss, han.accuracy, merge_summary_op], feed_dict=feed_dict)

        time_elapsed = time.time() - tic # end time for step
 
        if is_training:
          print("Training || CurrentEpoch: {} || GlobalStep: {} || ({} sec/step) || Loss {:g} || Accuracy {:g}".format(epoch+1, step, time_elapsed, loss, accuracy))

        if step % FLAGS.log_summaries_every == 0:
          train_summary_writer.add_summary(summaries, step)
          print("Saved model summaries to {}\n".format(os.path.join(paths.SUMMARY_DIR, FLAGS.run_type)))

        if step % FLAGS.checkpoint_every == 0:
          chkpt_path = saver.save(sess,
                                  os.path.join(paths.CHECKPOINT_DIR, "han"),
                                  global_step=step)
          print("Saved model checkpoint to {}\n".format(chkpt_path))
      # end

      # Generate batches
      imdb = IMDB(action="fetch")
      x_train, y_train, docsize_train, sent_size_train = imdb.get_data(type_=FLAGS.run_type)

      # Training loop. For each batch...
      for epoch, batch in imdb.get_batch_iter(data=list(zip(x_train, y_train, docsize_train, sent_size_train)),
                                              batch_size=FLAGS.batch_size,
                                              num_epochs=FLAGS.num_epochs):

        x_batch, y_batch, docsize, sent_size = zip(*batch)

        train_step(epoch=epoch,
                   x_batch=x_batch,
                   y_batch=y_batch,
                   docsize=docsize,
                   sent_size=sent_size,
                   is_training=True)

    sess.close()