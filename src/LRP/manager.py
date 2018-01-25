import datetime
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf

import data_helpers
from clover_lib import *

PATH_SPLIT2_TRAINED = "C:\\work\\Code\\PythonControversy\\src\\LRP\\runs\\1516259569\\checkpoints\\model-900"
PATH_FILTER_SHORT = "C:\work\Code\PythonControversy\src\LRP\\runs\\1516350455\checkpoints\model-900"
PATH_CLEAN_DATA = "C:\work\Code\PythonControversy\src\LRP\\runs\\1516424273\checkpoints\model-900"

PATH_SPLIT3 = "C:\work\Code\PythonControversy\src\LRP\\runs\\1516597480\checkpoints\model-200"
PATH_UNKNOWN = "C:\\work\\Code\\PythonControversy\\src\\LRP\\runs\\1516167942\\checkpoints\\model-500"

class Manager():
    def __init__(self, num_checkpoints, dropout_keep_prob,
                 batch_size, num_epochs, evaluate_every, checkpoint_every, initW):
        self.num_checkpoints = num_checkpoints
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.num_epochs=num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.initW = initW

    #
    def show_lrp(self, sess, lrp_manager, x, y, text):
        saver = tf.train.Saver(tf.global_variables())
        print("Restoring model")
        model_path = "C:\work\Code\PythonControversy\src\LRP\\runs\\1516655846\checkpoints\model-60"
        model_path = PATH_SPLIT3
        saver.restore(sess, model_path)

        feed_dict = {
            lrp_manager.cnn.input_x: x[:10],
            lrp_manager.cnn.input_y: y[:10],
            lrp_manager.cnn.dropout_keep_prob: 1.0
        }

        loss, accuracy = sess.run(
            [lrp_manager.cnn.loss, lrp_manager.cnn.accuracy],
            feed_dict)

        print("loss:{} accruacy:{}".format(loss, accuracy))

        # [self.batch_size, self.sequence_length
        r_t = lrp_manager.run(feed_dict)
        for i, batch in enumerate(r_t):
            print("-----------------------")
            if np.argmax(y[i]) == 0:
                continue
            tokens = text[i].split(" ")
            print("{}: {}".format(len(tokens), text[i]))
            indice = np.nonzero(batch)
            print(indice)
            for index in np.nditer(indice):
                if batch[index] > 0.1:
                    print("{} {} {}".format(index, tokens[index], batch[index]))

    def collect_correct(self, sess, cnn, x, y):
        feed_dict = {
            cnn.input_x: x,
            cnn.input_y: y,
            cnn.dropout_keep_prob: 1.0
        }

        [prediction] = sess.run(
            [cnn.predictions,],
            feed_dict)

        correct_x = []
        correct_y = []
        y_idx = np.argmax(y, axis=1)
        for i, _ in enumerate(x):
            if prediction[i] == y_idx[i]:
                correct_x.append(x[i])
                correct_y.append(y[i])
        return correct_x, correct_y

    # test accuracy drop after removing tokens
    def word_removing(self, sess, lrp_manager, x, y):
        saver = tf.train.Saver(tf.global_variables())
        model_path = PATH_CLEAN_DATA
        saver.restore(sess, model_path)

        correct_x, correct_y = self.collect_correct(sess, lrp_manager.cnn, x, y)

        def top_k_relevance(x, y):
            feed_dict = {
                lrp_manager.cnn.input_x: x,
                lrp_manager.cnn.input_y: y,
                lrp_manager.cnn.dropout_keep_prob: 1.0
            }
            #r_t = pickle.load(open("r_t.pickle", "rb"))
            r_t = lrp_manager.run(feed_dict)
            pickle.dump(r_t, open("r_t.pickle", "wb"))
            res = []
            for i, batch in enumerate(r_t):
                indice = np.flip(np.argsort(batch),0)
                res.append(indice)
            return res

        def get_precision(x,y):
            feed_dict = {
                lrp_manager.cnn.input_x: x,
                lrp_manager.cnn.input_y: y,
                lrp_manager.cnn.dropout_keep_prob: 1.0
            }
            [accuracy] = sess.run(
                [lrp_manager.cnn.accuracy, ],
                feed_dict)
            return accuracy

        def remove(source, rem_list):
            res = []
            for (i, item) in enumerate(source):
                if i not in rem_list:
                    res.append(item)
                else:
                    res.append(0)

            return np.array(res)

        relevance_list = top_k_relevance(correct_x, correct_y)

        for n_word_remove in range(0, 100, 10):
            rand_x = []
            rand_y = []
            lrp_x = []
            lrp_y = []
            for i, x in enumerate(correct_x):
                relevance = relevance_list[i]
                lrp_x.append(remove(x, relevance[:n_word_remove]))
                lrp_y.append(correct_y[i])
                real_len = max(relevance)
                rand_x.append(remove(x, np.random.choice(real_len, n_word_remove, replace=False)))
                rand_y.append(correct_y[i])

            print("Remove {}  LRP:{} Random:{}".format(
                n_word_remove,
                get_precision(lrp_x,lrp_y),
                get_precision(rand_x, rand_y)))


    def train(self, sess, cnn, x_train, x_dev, y_train, y_dev):
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(cnn.W.assign(self.initW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: self.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("Train {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("Dev {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % self.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % self.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

    def heatmap(self, sess, lrp_manager, test_data, answer, vocab_processor):
        saver = tf.train.Saver(tf.global_variables())
        model_path = PATH_CLEAN_DATA
        saver.restore(sess, model_path)
        def transform(text):
            return list(vocab_processor.transform(text))
        links, list_test_text = zip(*test_data)
        list_test_text = list([data_helpers.clean_str(x) for x in list_test_text])
        x_test = np.array(transform(list_test_text))
        y = np.array([[0,1]] * len(list_test_text))
        list_test_text = list(vocab_processor.reverse(x_test))

        feed_dict = {
            lrp_manager.cnn.input_x: x_test,
            lrp_manager.cnn.input_y: y,
            lrp_manager.cnn.dropout_keep_prob: 1.0
        }

        r_t = pickle.load(open("r_t.pickle", "rb"))
        #r_t = lrp_manager.run(feed_dict)
        #pickle.dump(r_t, open("r_t.pickle", "wb"))

        heatmaps = []
        for i, batch in enumerate(r_t):
            heatmap = []
            text_tokens = list_test_text[i].split(" ")
            for raw_index, value in np.ndenumerate(batch):
                index = raw_index[0]
                heatmap.append((text_tokens[index], value))
            heatmaps.append((answer[i], heatmap))
        pickle.dump(heatmaps, open("heatmap.pickle","wb"))

    def train_and_phrase(self, sess, lrp_manager, test_data, answer, vocab_processor):
        split2_frame = "C:\work\Code\PythonControversy\src\LRP\\runs\\1516655846\checkpoints\model-{}"
        def transform(text):
            return list(vocab_processor.transform(text))
        links, list_test_text = zip(*test_data)
        list_test_text = list([data_helpers.clean_str(x) for x in list_test_text])
        x_test = np.array(transform(list_test_text))
        y = np.array([[0,1]] * len(list_test_text))
        list_test_text = list(vocab_processor.reverse(x_test))
        def get_precision(x,y):
            feed_dict = {
                lrp_manager.cnn.input_x: x,
                lrp_manager.cnn.input_y: y,
                lrp_manager.cnn.dropout_keep_prob: 1.0
            }
            [accuracy] = sess.run(
                [lrp_manager.cnn.accuracy, ],
                feed_dict)
            return accuracy


        summary = []
        for progress in range(10, 150, 10):
            path = split2_frame.format(progress)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, path)

            feed_dict = {
                lrp_manager.cnn.input_x: x_test,
                lrp_manager.cnn.input_y: y,
                lrp_manager.cnn.dropout_keep_prob: 1.0
            }
            accuracy = get_precision(x_test, y)
            print("y/n Accuracy : ", end="")
            print(accuracy)

            def collect_correct_indice(sess, cnn, x, y):
                feed_dict = {
                    cnn.input_x: x,
                    cnn.input_y: y,
                    cnn.dropout_keep_prob: 1.0
                }

                [prediction] = sess.run(
                    [cnn.predictions, ],
                    feed_dict)

                indice = []
                y_idx = np.argmax(y, axis=1)
                for i, _ in enumerate(x):
                    if prediction[i] == y_idx[i]:
                        indice.append(i)
                return indice

            correct_indice = collect_correct_indice(sess, lrp_manager.cnn, x_test, y)

            r_t = lrp_manager.run(feed_dict)
            count = FailCounter()
            k = 10 # candiate to select
            for i, batch in enumerate(r_t):
                if i not in correct_indice:
                    continue
                if answer[i] is None:
                    continue
                phrase_len = len(answer[i].split(" "))
                candidates = []
                for raw_index, value in np.ndenumerate(batch):
                    index = raw_index[0]
                    if index + phrase_len >= batch.shape[0]:
                        break
                    assert (batch[index] == value)
                    window = batch[index:index+phrase_len]
                    s = sum(window)
                    candidates.append((s, index))
                candidates.sort(key=lambda x:x[0], reverse=True)
                answer_tokens = set(answer[i].lower().split(" "))
                text_tokens = list_test_text[i].split(" ")
                match = False
                for (value, last_index) in candidates[:k]:
                    end = last_index + 1
                    begin = end - phrase_len
                    sys_answer = text_tokens[begin:end]
                    if set(sys_answer) == answer_tokens:
                        match = True
                if match:
                    count.suc()
                else:
                    count.fail()
            print("Precision : {}".format(count.precision()))
            summary.append((progress, accuracy, count.precision() ))

        for progress, accuracy, precion in summary:
            print("{}\t{}\t{}\t".format(progress, accuracy, precion))

    # model_path = "C:\\work\\Code\\PythonControversy\\src\\LRP\\runs\\1516167942\\checkpoints\\model-500"


    # answer : list of tokens
    def test_phrase(self, sess, lrp_manager, test_data, answer, vocab_processor):
        saver = tf.train.Saver(tf.global_variables())
        #model_path = "C:\\work\\Code\\PythonControversy\\src\\LRP\\runs\\1516167942\\checkpoints\\model-500"
        model_path = PATH_SPLIT3
        saver.restore(sess, model_path)

        def transform(text):
            return list(vocab_processor.transform(text))
        links, list_test_text = zip(*test_data)
        list_test_text = list([data_helpers.clean_str(x) for x in list_test_text])
        x_test = np.array(transform(list_test_text))
        y = np.array([[0,1]] * len(list_test_text))
        list_test_text = list(vocab_processor.reverse(x_test))

        def get_precision(x,y):
            feed_dict = {
                lrp_manager.cnn.input_x: x,
                lrp_manager.cnn.input_y: y,
                lrp_manager.cnn.dropout_keep_prob: 1.0
            }
            [accuracy] = sess.run(
                [lrp_manager.cnn.accuracy, ],
                feed_dict)
            return accuracy
        accuracy = get_precision(x_test, y)
        print("Accuracy : ", end="")
        print(accuracy)

        def collect_correct_indice(sess, cnn, x, y):
            feed_dict = {
                cnn.input_x: x,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0
            }

            [prediction] = sess.run(
                [cnn.predictions, ],
                feed_dict)

            indice = []
            y_idx = np.argmax(y, axis=1)
            for i, _ in enumerate(x):
                if prediction[i] == y_idx[i]:
                    indice.append(i)
            return indice

        correct_indice = collect_correct_indice(sess, lrp_manager.cnn, x_test, y)

        feed_dict = {
            lrp_manager.cnn.input_x: x_test,
            lrp_manager.cnn.input_y: y,
            lrp_manager.cnn.dropout_keep_prob: 1.0
        }

        #r_t = pickle.load(open("r_t.pickle", "rb"))
        r_t = lrp_manager.run(feed_dict)
        pickle.dump(r_t, open("r_t.pickle", "wb"))

        res = []
        count = FailCounter()
        rand_count = FailCounter()
        k = 10 # candiate to select
        for i, batch in enumerate(r_t):
            #print("--------------")
            if i not in correct_indice:
                #print("Wrong")
                continue
            if answer[i] is None:
                #print("No answer in text")
                continue
            phrase_len = len(answer[i].split(" "))
            candidates = []
            for raw_index, value in np.ndenumerate(batch):
                index = raw_index[0]
                if index + phrase_len >= batch.shape[0]:
                    break
                assert (batch[index] == value)
                window = batch[index:index+phrase_len]
                s = sum(window)
                candidates.append((s, index))
            candidates.sort(key=lambda x:x[0], reverse=True)

            print("Collect: "+answer[i])
            answer_tokens = set(answer[i].lower().split(" "))
            text_tokens = list_test_text[i].split(" ")
            match = False
            rand_match = False

            def get_text(begin,end):
                if begin < 0 :
                    begin =0
                return " ".join(text_tokens[begin:end])

            for (value, last_index) in candidates[:k]:
                end = last_index + 1
                begin = end - phrase_len
                sys_answer = text_tokens[begin:end]
                print("{}-{} /{}: {}[{}]{}".format(begin, end, value, get_text(begin-3, begin), " ".join(sys_answer), get_text(end, end+3)) )
                if set(sys_answer) == answer_tokens:
                    match = True

                rand_begin = random.randint(0, 500)
                rand_end = rand_begin + phrase_len
                rand_answer = text_tokens[rand_begin:rand_end]
                if set(rand_answer) == answer_tokens:
                    rand_match = True
            if match:
                count.suc()
            else:
                count.fail()
            if rand_match:
                rand_count.suc()
            else:
                rand_count.fail()
            #print("")

            max_i = np.argmax(batch, axis=0)
            #print("Max : {} at {} ({})".format(batch[max_i], max_i, text_tokens[max_i]))

        print("Precision : {}".format(count.precision()))
        print("Precision[Random] : {}".format(rand_count.precision()))
        return res
