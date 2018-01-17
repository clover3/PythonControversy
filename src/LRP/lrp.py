import numpy as np

def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)


class LRPManager(object):
    def __init__(self, sess, cnn, sequence_length, num_classes, vocab_size, batch_size,
      embedding_size, filter_sizes, num_filters):
        self.sess = sess
        self.cnn = cnn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_filters_total = num_filters * len(filter_sizes)

        # trained parameters


    def eval_r(self, scores, predictions, dense_W, dense_b, pooledArg, maxPoolOutput, word_emb, filters):
        print("evaluating reverse relevance")
        # Update r_score
        print_shape("self.score:", scores)
        input_batch_size, num_classes = scores.shape
        r_k = np.zeros(shape=[input_batch_size, self.num_classes])
        for i in range(input_batch_size):
            idx = predictions[i]
            r_k[i, idx] = scores[i, idx]
        print_shape("dense_W:", dense_W)
        print_shape("dense_b:", dense_b)

        # maxPoolOutput : [batch , filter_size]
        z_jk = np.zeros(shape=[input_batch_size, self.num_filters_total, self.num_classes])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                for k in range(self.num_classes):
                    z_jk[batch, j, k] = maxPoolOutput[batch, j] * dense_W[j, k] + (dense_b[k]) / self.num_filters_total

        z_jk_sum = np.sum(z_jk, axis=1) # [batch_size * num_class]
        r_jk = np.zeros(shape=[input_batch_size, self.num_filters_total, self.num_classes])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                for k in range(self.num_classes):
                    r_jk[batch, j, k] = z_jk[batch, j, k] * r_k[batch, k] / z_jk_sum[batch, k]

        r_j = np.sum(r_jk, axis=2) # [ batch * num_filters_total)

        r_jt = np.zeros([self.batch_size, self.num_filters_total, self.sequence_length])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                t = pooledArg[batch, 0, j]
                r_jt[batch, j, t] = r_j[batch, j]
        print("getting z[j,t,tau]")
        z_jttau = np.zeros([self.batch_size, self.num_filters_total, self.sequence_length, max(self.filter_sizes)])
        j = 0
        #word_emb : [batch, sequence, dim]
        for filter_W, filter_b in filters:
            # filter_W : [filter_size, embedding_size, 1, num_filters]
            filter_size, embedding_size, _, num_filters = filter_W.shape
            for filter_idx in range(filter_size):
                for batch in range(input_batch_size):
                    for t in range(self.sequence_length - filter_size + 1):
                        if r_jt[batch, j, t] == 0 :
                            continue
                        for tau in range(filter_size):
                            x = word_emb[batch, t+tau, :] #
                            a = filter_W[tau, :, 0, filter_idx]
                            z_jttau[batch, j, t, tau] =  np.dot(x,a) + (filter_b[filter_idx])/(filter_size)

                j += 1
        print("getting r[t]")
        z_jttau_sum = z_jttau.sum(axis=3) # [batch_size * num_filters_total]
        # char at t to filter j
        r_tj = np.zeros([self.batch_size, self.sequence_length, self.num_filters_total])
        for batch in range(input_batch_size):
            j = 0
            for filter_W, filter_b in filters:
                # filter_W : [filter_size, embedding_size, 1, num_filters]
                filter_size, embedding_size, _, num_filters = filter_W.shape
                for filter_idx in range(filter_size):
                    for t in range(self.sequence_length - filter_size + 1):
                        if r_jt[batch, j,t] == 0:
                            continue
                        for tau in range(filter_size):
                            r_tj[batch, t+tau, j] = z_jttau[batch, j, t, tau] / z_jttau_sum[batch, j, t] * r_jt[batch, j,t]

                    j+= 1
        print(r_tj)
        return r_tj

    def run(self, feed_dict):
        predictions, scores, \
        dense_W, dense_b, filters, \
        word_emb, pooledArg, maxPoolOutput = self.sess.run(
            [self.cnn.predictions, self.cnn.scores,
             self.cnn.dense_W, self.cnn.dense_b, self.cnn.filters,
             self.cnn.embedded_chars , self.cnn.pooledArg, self.cnn.h_pool_flat,
             ],
            feed_dict)
        return self.eval_r(
            scores=scores,
            predictions=predictions,
            dense_W=dense_W,
            dense_b=dense_b,
            filters=filters,
            word_emb=word_emb,
            pooledArg=pooledArg,
            maxPoolOutput=maxPoolOutput
        )

