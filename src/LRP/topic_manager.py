import tensorflow as tf
import datetime
import data_helpers
import os
import time
from TopicCNN import TopicCNN
import numpy as np
from tensorflow.contrib import learn
import pickle
import collections
import math
from nltk.tokenize import wordpunct_tokenize
from statistics import mean


class BM25:
    def __init__(self):
        self.k1 = 1.2
        self.k2 = 10
        self.b = 0.75
        self.avgDocLen = 1000

    def tokenize(self, str):
        return wordpunct_tokenize(str)

    def relevance(self, text, topics, split_no):
        bm_pickle = "bm25_{}.pickle".format(split_no)
        if os.path.exists(bm_pickle):
            return pickle.load(open(bm_pickle, "rb"))
        phrases = [topic.split(" ") for topic in topics]

        tokened_corpus = [self.tokenize(s) for s in text]
        self.n_docs = len(tokened_corpus)
        self.df = collections.Counter()
        print("Evaluating BM25")
        self.doc_sparse = []
        voca = collections.Counter()
        for tokens in tokened_corpus:
            doc_sparse = collections.Counter()
            for token in tokens:
                voca[token] += 1
                doc_sparse[token] += 1
            for key in doc_sparse:
                self.df[key] += 1
            self.doc_sparse.append(doc_sparse)
        self.total_words = sum(voca.values())

        def bm25(doc_sparse, phrase):
            doclen = sum(doc_sparse.values())
            K = self.k1 * ((1 - self.b) + self.b * doclen / self.avgDocLen)

            def get_tf(word):
                if word in doc_sparse:
                    return doc_sparse[word]
                else:
                    return 0

            score = 0
            for word in phrase:
                idf = math.log((self.n_docs - self.df[word] + 0.5) / (self.df[word] + 0.5))
                tf = get_tf(word)
                doc_side = tf * (1 + self.k1) / (tf + self.k1 * K)
                query_side = 1 * (1 + self.k2) / (1 + self.k2)
                score = score + idf * doc_side * query_side
            return score

        score_list = []
        for doc in self.doc_sparse:
            scores = []
            for phrase in phrases:
                score = bm25(doc, phrase)
                scores.append(score)
            score_list.append(np.array(scores))
        result = np.array(score_list)
        pickle.dump(result, open(bm_pickle, "wb"))
        return result

class TopicManager():
    def __init__(self, num_checkpoints, dropout_keep_prob,
                 batch_size, num_epochs, evaluate_every, checkpoint_every):
        self.num_checkpoints = num_checkpoints
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.num_epochs=num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every


    def train(self, sess, x_text, y, split_no, FLAGS):
        max_document_length = 5000
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        pickle.dump(vocab_processor, open("vocabproc{}.pickle".format(split_no), "wb"))
        #vocab_processor = pickle.load(open("vocabproc{}.pickle".format(split_no), "rb"))

        topics = pickle.load(open("phrase3000_{}.pickle".format(split_no), "rb"))
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        y = np.array(y)
        bm25 = BM25()

        topics = bm25.relevance(x_text, topics, split_no)

        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        t_shuffled = topics[shuffle_indices]
        # Build vocabulary

        text_x_shuffled = []
        for index in np.nditer(shuffle_indices):
            text_x_shuffled.append(x_text[index])
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))


        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        t_train, t_dev = t_shuffled[:dev_sample_index], t_shuffled[dev_sample_index:]
        del x, y, x_shuffled, y_shuffled

        self.initW = data_helpers.load_word2vec(vocab_processor, FLAGS.embedding_dim)
        cnn = TopicCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            batch_size=FLAGS.batch_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_topics=FLAGS.num_topics,
            topic_lambda=FLAGS.topic_lambda,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        self.train_nn(sess, cnn, x_train, x_dev, t_train, y_train, y_dev, t_dev)

    def train_nn(self, sess, cnn, x_train, x_dev, t_train, y_train, y_dev, t_dev):
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)

        grads_W = optimizer.compute_gradients(cnn.tp_pred_loss)
        train_op1 = optimizer.apply_gradients(grads_W, global_step=global_step)
        optimizer1 = tf.train.AdamOptimizer(1e-3)
        grad_topic = optimizer1.compute_gradients(cnn.tp_loss)
        train_op2 = optimizer1.apply_gradients(grad_topic, global_step=global_step)
        optimizer2 = tf.train.AdamOptimizer(1e-3)
        grad_global = optimizer2.compute_gradients(cnn.loss)
        train_op3 = optimizer2.apply_gradients(grad_global, global_step=global_step)

        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        self.state = 0
        sess.run(tf.global_variables_initializer())
        sess.run(cnn.W.assign(self.initW))

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

        def state_change(train_info_list):
            info = [float(sum(col))/len(col) for col in zip(*train_info_list)]
            topic_prediction_loss = info[7]
            topic_loss = info[4]
            if self.state == 0 and topic_prediction_loss < 0.02:
                print("Now optimizing tp loss")
                self.train_op = train_op2
                self.state = 1
            if self.state == 1 and topic_loss < 0.02:
                print("Now optimizing global loss")
                self.train_op = train_op3
                self.state = 2

        def train_step(x_batch, y_batch, t_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_topic: t_batch,
                cnn.dropout_keep_prob: self.dropout_keep_prob
            }
            _, step, loss, accuracy, tp_prec, tp_acc, \
            nn_pred_loss, tp_pred_loss,\
            pred_loss, l2_loss, tp_loss = sess.run(
                [self.train_op, global_step, cnn.loss, cnn.accuracy, cnn.tp_prec, cnn.tp_acc,
                 cnn.nn_pred_loss, cnn.tp_pred_loss,
                 cnn.pred_loss, cnn.l2_loss, cnn.tp_loss
                 ],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("\rTrain : step {}, loss {}, acc {}".format(step, loss_str, accuracy))
            return accuracy, loss, pred_loss, l2_loss, tp_loss, tp_prec, nn_pred_loss, tp_pred_loss, tp_acc

        def dev_step(x_dev, y_dev, t_dev):
            """
            Evaluates model on a dev set
            """
            batches = data_helpers.batch_iter(
                list(zip(x_dev, y_dev, t_dev)), self.batch_size, 1)
            accuracy_l = []
            loss_l = []
            for batch in batches:
                x_batch, y_batch, t_batch = zip(*batch)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.input_topic: t_batch,
                    cnn.dropout_keep_prob: 1.0,
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                accuracy_l.append(accuracy * len(y_batch))
                loss_l.append(loss * len(y_batch))
            accuracy = sum(accuracy_l) / len(y_dev)
            loss = sum(loss_l) / len(y_dev)
            time_str = datetime.datetime.now().isoformat()
            return accuracy, loss

        def train_info_str(info_list, current_step):
            info = [float(sum(col))/len(col) for col in zip(*info_list)]
            info_str = "loss:{0:.4f} ".format(info[1]) \
                       + "pred={0:.4f}".format(info[2]) \
                        + "[nn:{0:.4f},tp:{1:.4f}] ".format(info[6], info[7]) \
                        + "l2:{0:.4f} ".format(info[3]) \
                        + "tp_loss:{0:.5f} ".format(info[4]) \
                        + "tp_prec:{0:.6f} ".format(info[5]) \
                        + "tp_acc:{0:.2f} ".format(info[8]) \
                        + "acc:{0:.2f}".format(info[0])
            #loss_str = "loss:{0:.4f} pred={1:.4f}[{5:.4f},{6:.4f}] l2={2:.4f} topic={3:.5f} tp_prec={4:.2f} " \
            #           "tp_acc{7:0.5".format(info[1], info[2], info[3], info[4], info[5], info[6], info[7])
            return "Train : step {}, {}, acc {}".format(current_step, info_str, info[0])



        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train, t_train)), self.batch_size, self.num_epochs)
        # Training loop. For each batch...
        num_batches_per_epoch = int((len(y_train) - 1) / self.batch_size) + 1
        evaluate_batch = self.evaluate_every * len(y_train) / self.batch_size
        print("batch per epoch {}".format(num_batches_per_epoch))
        train_info_list = []
        for batch in batches:
            x_batch, y_batch, t_batch = zip(*batch)
            train_info = train_step(x_batch, y_batch, t_batch)
            train_info_list.append(train_info)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % self.evaluate_every == 0:
                train_str = train_info_str(train_info_list, current_step)
                #state_change(train_info_list)
                train_info_list = []

                dev_acc, dev_loss = dev_step(x_dev, y_dev, t_dev)
                print("{} dev_acc={} dev_loss={}".format(train_str, dev_acc, dev_loss))
            if current_step % self.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        accuracy = dev_step(x_dev, y_dev, t_dev)
        return accuracy

