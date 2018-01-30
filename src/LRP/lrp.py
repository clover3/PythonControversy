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
        input_batch_size, num_classes = scores.shape
        r_k = np.zeros(shape=[input_batch_size, self.num_classes])
        for i in range(input_batch_size):
            idx = predictions[i]
            for c in range(num_classes):
                if 1 == c:
                    r_k[i, c] = scores[i, c]
                else:
                    r_k[i, c] = -scores[i, c]
        # maxPoolOutput : [batch , filter_size]
        z_jk = np.zeros(shape=[input_batch_size, self.num_filters_total, self.num_classes])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                for k in range(self.num_classes):
                    z_jk[batch, j, k] = maxPoolOutput[batch, j] * dense_W[j, k] + (dense_b[k]+0.01) / self.num_filters_total

        z_jk_sum = np.sum(z_jk, axis=1) # [batch_size * num_class]
        r_jk = np.zeros(shape=[input_batch_size, self.num_filters_total, self.num_classes])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                for k in range(self.num_classes):
                    r_jk[batch, j, k] = z_jk[batch, j, k] * r_k[batch, k] / z_jk_sum[batch, k]

        r_j = np.sum(r_jk, axis=2) # [ batch * num_filters_total)
        assert( np.count_nonzero(r_j[0, :]) == self.num_filters_total)

        r_jt = np.zeros([input_batch_size, self.num_filters_total, self.sequence_length])
        for batch in range(input_batch_size):
            for j in range(self.num_filters_total):
                t = pooledArg[batch, 0, j]
                r_jt[batch, j, t] = r_j[batch, j]
        assert( np.count_nonzero(r_jt.sum(axis=2).sum(axis=1)) == input_batch_size)
        z_jttau = np.zeros([input_batch_size, self.num_filters_total, self.sequence_length, max(self.filter_sizes)])
        j = 0
        #word_emb : [batch, sequence, dim]
        for filter_W, filter_b in filters:
            # filter_W : [filter_size, embedding_size, 1, num_filters]
            filter_size, embedding_size, _, num_filters = filter_W.shape
            for filter_idx in range(num_filters):
                for batch in range(input_batch_size):
                    for t in range(self.sequence_length - filter_size + 1):
                        if r_jt[batch, j, t] == 0 :
                            continue
                        for tau in range(filter_size):
                            x = word_emb[batch, t+tau, :] #
                            a = filter_W[tau, :, 0, filter_idx]
                            assert(np.dot(x,a).nonzero())
                            z_jttau[batch, j, t, tau] = np.dot(x,a) + (filter_b[filter_idx]+0.01)/(filter_size)

                j += 1
        z_jttau_sum = z_jttau.sum(axis=3) # [batch_size * num_filters_total * sequence_length]
        assert (np.count_nonzero(z_jttau_sum.sum(axis=2).sum(axis=1)) == input_batch_size)
        # char at t to filter j
        r_tj = np.zeros([input_batch_size, self.sequence_length, self.num_filters_total])
        for batch in range(input_batch_size):
            j = 0
            for filter_W, filter_b in filters:
                # filter_W : [filter_size, embedding_size, 1, num_filters]
                filter_size, embedding_size, _, num_filters = filter_W.shape
                for filter_idx in range(num_filters):
                    for t in range(self.sequence_length - filter_size + 1):
                        if r_jt[batch, j, t] == 0:
                            continue
                        portion_sum =0
                        for tau in range(filter_size):
                            portion = z_jttau[batch, j, t, tau] / z_jttau_sum[batch, j, t]
                            r_tj[batch, t+tau, j] = r_tj[batch, t+tau, j] + portion * r_jt[batch, j, t]
                            portion_sum = portion_sum + portion
                        assert( 1-portion_sum < 0.00001)
                    j+= 1
        r_t = r_tj.sum(axis=2) # [ input_batch_size , sequence_length]
        assert(np.count_nonzero(r_t.sum(axis=1)) == input_batch_size)
        return r_t

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

