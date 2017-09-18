"""
@author: Michael Guarino
TODO: run this
"""

import tensorflow as tf
import numpy as np
import os
import logging
import datetime
from dataProcessing import IMDB
from han_gru import HAN_GRU
from utils import prjPaths

print("current version of tf:{}".format(tf.__version__))

# model configuration
tf.flags.DEFINE_string("run_type", "train",
                       "enter train or test to specify run_type (default: train)")
tf.flags.DEFINE_integer("embedding_dim", 100,
                        "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("batch_size", 4,
                        "Batch Size (default: 4)")
tf.flags.DEFINE_integer("num_epochs", 100,
                        "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps")
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

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

assert(FLAGS.run_type == "train" or FLAGS.run_type == "test" or FLAGS.run_type == "restore_train"), "please specify valid run_type"

paths = prjPaths()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
currentTime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
logFileName = os.path.join(paths.LOGS_DIR, "HAN_GRU_TxtClassification_{}.log".format(currentTime))

fileHandler = logging.FileHandler(logFileName)
fileHandler.setLevel(logging.ERROR)
fileHandler.setFormatter(formatter)

logger.addHandler(fileHandler)

print("Loading data...\n")

if not IMDB.csvExist():
    imdb = IMDB(action="create")
    imdb.createManager()
    x_train, y_train, x_test, y_test = imdb.partitionManager(type="han")
else:
    imdb = IMDB()
    x_train, y_train, x_test, y_test = imdb.partitionManager(type="han")

if FLAGS.run_type == "train":
    print("Training...\n")
    # create new graph set as default
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        session_conf.gpu_options.allocator_type = "BFC"
        # create new session set it as default
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # create new cnn model
            han = HAN_GRU(max_seq_len=imdb.max_seq_len,
                      max_sent_len=imdb.max_sent_len,
                      num_classes=len(y_test[0]),
                      vocab_size=imdb.vocab_size,
                      embedding_size=FLAGS.embedding_dim,
                      max_grad_norm=FLAGS.max_grad_norm,
                      dropout_keep_proba=FLAGS.dropout_keep_proba,
                      learning_rate=FLAGS.learning_rate)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(han.loss, tvars),
                                                        han.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(han.learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 name="train_op",
                                                 global_step=global_step)

            # checkpoint model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x, y_batch):
                x_batch, docsize, sent_size = x

                feed_dict = {
                    han.input_x: x_batch,
                    han.input_y: y_batch,
                    han.sentence_lengths: docsize,
                    han.word_lengths: sent_size,
                    han.is_training: True
                }

                _, step, loss, accuracy = sess.run([train_op, global_step, han.loss, han.accuracy], feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # end

            # Generate batches
            batches = imdb.get_batch_iter(list(zip(x_train, y_train)),
                                          FLAGS.batch_size,
                                          FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, docsize, sent_size = imdb.hanformater(inputs=x_batch)
                x = [x_batch, docsize, sent_size]
                train_step(x, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    chkpt_path = saver.save(sess,
                                            paths.CHECKPOINTS_HAN_GRU+"han_gru",
                                            global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(chkpt_path))
        sess.close()

elif FLAGS.run_type == "test":
    print("Testing...\n")

    checkpoint_file = tf.train.latest_checkpoint(paths.CHECKPOINTS_HAN_GRU)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables

            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            word_lengths = graph.get_operation_by_name("word_lengths").outputs[0]
            sentence_lengths = graph.get_operation_by_name("sentence_lengths").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("classifier_output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = imdb.get_batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, docsize, sent_size = imdb.hanformater(inputs=x_batch)
                batch_predictions = sess.run(predictions, {input_x:x_batch, word_lengths:sent_size,
                                             sentence_lengths:docsize, is_training:True})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
        sess.close()

    print(all_predictions[0:2])
    all_predictions = imdb._oneHot(all_predictions)
    print(all_predictions[0:2])
    print("all_predictions shape:{}".format(all_predictions.shape))
    print(y_test[0:2])

    # Print accuracy if y_test is defined
    if y_test is not None:
        assert(len(y_test)==len(all_predictions)), "y_test and predictions should be same length"
        correct_predictions = [i for i in range(len(y_test)) if np.allclose(y_test[i], all_predictions[i])]
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(float(len(correct_predictions)/len(y_test))))
