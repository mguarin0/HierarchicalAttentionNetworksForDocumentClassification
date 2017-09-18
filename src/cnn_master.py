"""
@author: Michael Guarino adapted for Denny Britz https://github.com/dennybritz/cnn-text-classification-tf.git
TODO: currently training
"""

import tensorflow as tf
import numpy as np
import os
import logging
import datetime
from dataProcessing import IMDB
from cnn import CNN
from utils import prjPaths

print("current version of tf:{}".format(tf.__version__))

# model configuration
tf.flags.DEFINE_string("run_type", "train",
                       "enter train or test to specify run_type (default: train)")
tf.flags.DEFINE_integer("embedding_dim", 200,
                        "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_string("filter_sizes", "5,7,9,11",
                       "Comma-separated filter sizes (default: '5,7,9,11')")
tf.flags.DEFINE_integer("num_filters", 128,
                        "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 100,
                        "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 1000,
                        "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100,
                        "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 1,
                        "Number of checkpoints to store (default: 1)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

assert(FLAGS.run_type == "train" or FLAGS.run_type == "test"), "please specify valid run_type"

paths = prjPaths()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
currentTime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
logFileName = os.path.join(paths.LOGS_DIR, "CNN_TxtClassification_{}.log".format(currentTime))

fileHandler = logging.FileHandler(logFileName)
fileHandler.setLevel(logging.ERROR)
fileHandler.setFormatter(formatter)

logger.addHandler(fileHandler)

print("Loading data...\n")

if not IMDB.csvExist():
    imdb = IMDB(action="create")
    imdb.createManager()
    x_train, y_train, x_test, y_test = imdb.partitionManager(type="cnn")
else:
    imdb = IMDB()
    x_train, y_train, x_test, y_test = imdb.partitionManager(type="cnn")

if FLAGS.run_type == "train":
    print("Training...\n")

    # create new graph set as default
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)

        # create new session set it as default
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            # create new cnn model
            cnn = CNN(
                max_seq_len=imdb.max_seq_len,
                num_classes=len(y_test[0]),
                vocab_size=imdb.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=0.0)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # checkpoint model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # send find ops to sess.run
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],
                                                   feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # end

            # Generate batches
            batches = imdb.get_batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    chkpt_path = saver.save(sess, paths.CHECKPOINTS_CNN+"cnn", global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(chkpt_path))
        sess.close()

elif FLAGS.run_type == "test":
    print("Testing...\n")
    print(paths.CHECKPOINTS_CNN)

    checkpoint_file = tf.train.latest_checkpoint(paths.CHECKPOINTS_CNN)
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
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("classifier_output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = imdb.get_batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
        sess.close()

    print(all_predictions[0:1])
    all_predictions = imdb._oneHot(all_predictions)
    print(all_predictions[0:1])
    print(y_test[0:1])
    print("all_predictions shape:{}".format(all_predictions.shape))

    # Print accuracy if y_test is defined
    if y_test is not None:
        assert(len(y_test)==len(all_predictions)), "y_test and predictions should be same length"
        correct_predictions = [i for i in range(len(y_test)) if np.allclose(y_test[i], all_predictions[i])]
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(float(len(correct_predictions)/len(y_test))))
