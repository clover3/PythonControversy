import numpy as np


class LRPTopic(object):
    def __init__(self, sess, cnn, sequence_length, num_classes, batch_size, num_topics):
        self.sess = sess
        self.cnn = cnn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_topics = num_topics

        # trained parameters

    def eval_r(self, scores, dense_W, dense_b, topic_output):
        print("evaluating reverse relevance")
        # Update r_score
        input_batch_size, num_classes = scores.shape
        r_k = np.zeros(shape=[input_batch_size, self.num_classes])
        for i in range(input_batch_size):
            for c in range(num_classes):
                if 1 == c:
                    r_k[i, c] = scores[i, c]
                else:
                    r_k[i, c] = 0#-scores[i, c]
        # topic neurons : [batch , filter_size]
        print("Dense W shape:{}".format(dense_W.shape))
        z_sk = np.zeros(shape=[input_batch_size, self.num_topics, self.num_classes])
        for batch in range(input_batch_size):
            for s in range(self.num_topics):
                for k in range(self.num_classes):
                    z_sk[batch, s, k] = topic_output[batch, s] * dense_W[s, k] + (dense_b[k]+0.01) / self.num_topics

        z_sk_sum = np.sum(z_sk, axis=1) # [batch_size * num_classes]

        r_sk = np.zeros(shape=[input_batch_size, self.num_topics, self.num_classes])
        for batch in range(input_batch_size):
            for s in range(self.num_topics):
                for k in range(self.num_classes):
                    r_sk[batch, s, k] = z_sk[batch, s, k] * r_k[batch, k] / z_sk_sum[batch, k]

        r_s = np.sum(r_sk, axis=2) # [ batch * num_topics)
        return r_s

    def run(self, feed_dict):
        scores, input_topic, tp_loss,\
        dense_W, dense_b, topic_output,\
        accuracy, tp_accuracy = self.sess.run(
            [self.cnn.scores, self.cnn.input_topic, self.cnn.tp_loss,
             self.cnn.dense_W, self.cnn.dense_b, self.cnn.topic_prob,
             self.cnn.accuracy, self.cnn.tp_accuracy,
             ],
            feed_dict)
        print("Accuracy={} TpAccuracy={} Topic loss : {}".format(accuracy, tp_accuracy, tp_loss))
        return self.eval_r(
            scores=scores,
            dense_W=dense_W,
            dense_b=dense_b,
            topic_output=topic_output
        )

