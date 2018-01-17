import tensorflow as tf
import numpy as np

def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)



class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, batch_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        print(sequence_length)
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.filters = []

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        print_shape("emb_chars_exp:", self.embedded_chars_expanded)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        arg_list = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                print_shape("convol W:", W)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.filters.append((W,b))
                W_ = tf.reshape(W, [filter_size, embedding_size, num_filters])
                W_ = tf.tile(W_, [batch_size,1,1])
                print_shape("W_:", W_)
                W_ = tf.reshape(W_, [batch_size, filter_size, embedding_size, num_filters])
                # manual convolution
                z_ijt_list = [[]] * (sequence_length - filter_size + 1)
                for t in [0]: #range(sequence_length-filter_size):
                    text = self.embedded_chars[:, t:t+filter_size, :] #[batch, filter_size, dim]
                    print_shape("text:", text)
                    text_ = tf.tile(text, [1, 1, num_filters])
                    print_shape("text_:", text_)
                    text_ = tf.reshape(text_, [batch_size, filter_size, embedding_size, num_filters])
                    z_ = tf.multiply(text_, W_)
                    #for tau in range(filter_size):
                    #    z_itaujt = z_[:, tau, :, :]
                    #    z_itaujt = tf.reshape(z_itaujt, [batch_size, embedding_size, num_filters])
                    #    z_ijt_list[tau+t].append(z_itaujt) # z_[...} : [batch, embedding_size, num_filters]


                #z_ijt_list2 = []
                #for l in z_ijt_list:
                #    z_ijt = tf.reduce_sum(tf.stack(z_ijt_list, axis=3), axis=3) # [batch, embedding, num_filters, filter_size(not always) -> 1]
                #    z_ijt_list2.append(z_ijt)
                #z_ijt = tf.stack(z_ijt_list2, axis=3) # [batch, embedding, num_filters, sequence_length]

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print_shape("conv:", conv)
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
                print_shape("arg:", arg)
                print_shape("pooled:", pooled)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.pooledArg = tf.concat(arg_list, 2)
        print_shape("pooledArg:", self.pooledArg)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print_shape("h_pool:", self.h_pool)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print_shape("h_pool_flat:", self.h_pool_flat)
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.dense_W = W
            self.dense_b = b
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            print_shape("h_drop:", self.h_drop)
            print_shape("W:", W)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") # [batch, num_class]
            self.predictions = tf.argmax(self.scores, 1, name="predictions") # [batch , 1]

        with tf.name_scope("reverse"):
            print("Building reverse")
            # Update r_score
            print_shape("self.score:", self.scores)
            numbering = tf.range(0, batch_size, 1, dtype=tf.int64)
            s_prediction = tf.reshape(self.predictions, [batch_size,])
            indice = tf.stack([numbering, s_prediction], axis=1)
            print_shape("indice :", indice )
            values = tf.gather_nd(tf.reshape(self.scores, [batch_size, num_classes]), indice) # [batch,]
            print_shape("value:", values)
            print_shape("indice:", indice)
            r_k = tf.SparseTensor(indices=indice, values=values, dense_shape=[self.batch_size, num_classes])
            #
            x = tf.tile(self.h_pool_flat, [num_classes,1])
            x = tf.reshape(x, [num_classes, self.batch_size, num_filters_total])
            x = tf.transpose(x, [1,2,0])
            print_shape("x:", x)
            w_tile = tf.reshape(tf.tile(W, [self.batch_size,1]), [self.batch_size, num_filters_total, num_classes])
            xw = tf.multiply(x, w_tile)
            b_tile = tf.reshape(tf.tile(b, [self.batch_size*num_filters_total]), [self.batch_size, num_filters_total, num_classes])
            print_shape("b_tile:", b_tile)
            z = xw + b_tile
            print_shape("z:", z)
            z_sum = tf.reduce_sum(z, axis=1)
            z_sum_tile = tf.tile(z_sum, [num_filters_total,1])
            print_shape("z_sum_tile:", z_sum_tile)
            z_sum_tile = tf.reshape(z_sum_tile, [num_filters_total, self.batch_size, num_classes])
            z_sum_tile= tf.transpose(z_sum_tile, [1,0,2])

            r_k = tf.sparse_tensor_to_dense(r_k, 0)
            r_k_tile = tf.tile(r_k, [num_filters_total,1])
            r_k_tile = tf.reshape(r_k_tile, [num_filters_total, self.batch_size, num_classes])
            r_k_tile = tf.transpose(r_k_tile, [1,0,2])

            # r_jk = z_jk
            r_jk = tf.divide(tf.multiply(r_k_tile, z), z_sum_tile)
            print_shape("r_jk:", r_jk)
            print("Building reverse Done")

            r_j = tf.reduce_sum(r_jk, axis=2)
            print_shape("r_j:", r_j)
            argt_flat = tf.reshape(self.pooledArg, [self.batch_size, num_filters_total])
            print_shape("argt_flat:", argt_flat)
            filter_idx = tf.range(0, num_filters_total, 1, dtype=tf.int64)
            filter_idx = tf.reshape(tf.tile(filter_idx, [self.batch_size]), [self.batch_size, num_filters_total])
            print_shape("filter_idx:", filter_idx)

            batch_idx = tf.range(0, self.batch_size, 1, dtype=tf.int64) # [batch_size]
            batch_idx = tf.tile(batch_idx, [num_filters_total])
            batch_idx = tf.reshape(batch_idx, [num_filters_total, self.batch_size])
            batch_idx = tf.transpose(batch_idx, [1,0])

            indice = tf.stack([batch_idx, filter_idx, argt_flat], axis=2)
            indice = tf.reshape(indice, shape=[-1,3])
            print_shape("indice:", indice)
            values = tf.reshape(r_j, shape=[-1])
            r_jt = tf.SparseTensor(indices=indice, values=values, dense_shape=[self.batch_size, num_filters_total, sequence_length])
            #print_shape("r_jt:", r_jt)

            #exit(10)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
