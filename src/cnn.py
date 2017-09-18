"""
@author: Michael Guarino adapted for Denny Britz https://github.com/dennybritz/cnn-text-classification-tf.git
TODO: currently training
"""

import tensorflow as tf


class CNN:
    def __init__(self, max_seq_len, num_classes, vocab_size,
         embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

         # Input to network
         self.input_x = tf.placeholder(tf.int32, [None, max_seq_len], name="input_x")
         self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
         self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

         l2_loss = tf.constant(0.0)

         with tf.device("/gpu:0"), tf.name_scope("embedding_layer"):
             w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="w")
             self.embedded_chars = tf.nn.embedding_lookup(w, self.input_x)
             self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

         pooled_outputs = []
         for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_maxpool_layer_filter_{}".format(filter_size)):
                # conv layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",  # (narrow convolution)
                    name="conv")  # [1, (max_seq_len-filter_size)+1,1,1]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',  # (narrow convolution)
                    name="pool")  # [batch_size, 1, 1, num_filters]
                pooled_outputs.append(pooled)

         # feature vector
         num_filters_total = num_filters * len(filter_sizes)
         self.h_pool = tf.concat(pooled_outputs, 3)
         self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

         # Add dropout
         with tf.name_scope("dropout"):
              self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

         # Final (unnormalized) scores and predictions
         with tf.name_scope("classifier_output"):
             w = tf.get_variable(
                 "w",
                 shape=[num_filters_total, num_classes],
                 initializer=tf.contrib.layers.xavier_initializer())
             b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
             l2_loss += tf.nn.l2_loss(w)
             l2_loss += tf.nn.l2_loss(b)
             self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")
             self.predictions = tf.argmax(self.scores, 1, name="predictions")

         # CalculateMean cross-entropy loss
         with tf.name_scope("loss"):
             losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
             self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

         # Accuracy
         with tf.name_scope("accuracy"):
             correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
             self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
