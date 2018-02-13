import tensorflow as tf
import numpy as np


def get_precision(sess, cnn, x, y):
    feed_dict = {
        cnn.input_x: x,
        cnn.input_y: y,
        cnn.dropout_keep_prob: 1.0
    }
    [accuracy] = sess.run(
        [cnn.accuracy, ],
        feed_dict)
    return accuracy


def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)





class TopicCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_topics, num_classes, vocab_size, batch_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, topic_lambda=0.1):
        print("Building CNN")
        print("lambda: l2={} topic={}".format(l2_reg_lambda, topic_lambda))
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_topic = tf.placeholder(tf.float32, [None, num_topics], name="input_topic")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.filters = []

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"), tf.device('/device:GPU:0'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        arg_list = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.filters.append((W,b))

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                arg = tf.argmax(h, axis=1)
                arg_list.append(arg)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.pooledArg = tf.concat(arg_list, 2)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("topic"):
            W = tf.get_variable(
                "Wt",
                shape=[num_filters_total, num_topics],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_topics]), name="b")
            self.topic_out = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")  #
            self.topic_prob = tf.nn.sigmoid(self.topic_out)

        with tf.name_scope("bm25"):
            a = tf.get_variable(
                "a",
                shape=[num_topics,],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_topics]), name="b")
            self.input_topic_prob = tf.nn.sigmoid(tf.multiply(a,self.input_topic) + b)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_topics, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.dense_W = W
            self.dense_b = b
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.topic_drop = tf.nn.dropout(self.topic_prob, self.dropout_keep_prob)
            #self.scores = tf.nn.xw_plus_b(self.topic_drop, W, b, name="scores") # [batch, num_class]
            self.scores = tf.reduce_max(self.topic_prob, axis=1)
            self.max_idx = tf.argmax(self.topic_prob, output_type=tf.int64, axis=1)

            indice = tf.stack([tf.range(0,self.batch_size, dtype=tf.int64), self.max_idx], axis=1)
            self.tp_scores = tf.gather_nd(self.input_topic_prob, indice)
            print_shape("self.scores:", self.scores)

            def dd(scores):
                t1 = tf.constant(1.0, dtype=tf.float32, shape=[batch_size])
                return tf.stack([t1 - scores, tf.reshape(scores, shape=t1.shape)], axis=1)

            self.scores = dd(self.scores)
            self.tp_scores = dd(self.tp_scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions") # [batch , 1]
            print_shape("self.predictions:", self.predictions)
            self.tp_predictions = tf.argmax(self.tp_scores, 1, name="predictions")  # [batch , 1]
            print_shape("self.tp_predictions:", self.tp_predictions)

        def max_loss(base, target):
            return tf.reduce_max(tf.abs(target-base), axis=1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.nn_pred_loss = tf.reduce_mean(losses)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.tp_scores, labels=self.input_y)
            self.tp_pred_loss = tf.reduce_mean(losses)
            self.pred_loss = self.nn_pred_loss
            self.l2_loss = l2_loss
            z = tf.einsum("ab,bc->abc", self.topic_out, W)
            z_both = z[:,:,1]
            diff = tf.abs(self.topic_prob - self.input_topic_prob)

            z_sum = tf.reduce_sum(z_both, axis=1)
            unknown = tf.reduce_sum(tf.multiply(tf.abs(z_both), diff), axis=1) / z_sum
            print(unknown.shape)
            #self.tp_loss = tf.reduce_mean(unknown)
            print(self.tp_predictions.shape)
            print(self.predictions.shape)
            self.tp_loss = self.tp_pred_loss
            self.loss = self.pred_loss + l2_reg_lambda * self.l2_loss + topic_lambda * self.tp_loss

            self.top_topic_idx = self.max_idx
        def binary(tensor):
            return tf.cast(tensor + tf.constant(0.5, shape=[self.batch_size, num_topics]), tf.int32)

        def accuracy(label, predictions):
            correct_predictions = tf.equal(predictions, label)
            return tf.reduce_mean(tf.cast(correct_predictions, "float"))

        def precision(labels, predictions):
            a = tf.cast(labels, tf.bool)
            b = tf.cast(predictions, tf.bool)
            judge = tf.logical_not(tf.logical_and(tf.logical_xor(a,b), b))
            return tf.reduce_mean(tf.cast(judge, tf.float32))

        # Accuracy
        with tf.name_scope("accuracy"):
            self.accuracy = accuracy(self.predictions, tf.argmax(self.input_y, 1))
            self.tp_acc = accuracy(self.tp_predictions, tf.argmax(self.input_y, 1))
            self.tp_prec = precision(predictions=binary(self.topic_prob), labels=binary(self.input_topic_prob))
