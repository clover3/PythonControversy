from clover_lib import *
from model_common import *
from view import Logger
import os
import time


class ContrvWord(object):
    def __init__(self, config, sess):

        self.doc_max_word = config.doc_max_word
        self.nwords = config.nwords
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.optimizer = config.optimizer
        self.current_lr = config.init_lr
        self.is_test = config.is_test

        # document as set of words
        self.doc_word = tf.placeholder(tf.int32, [None, self.doc_max_word,], "doc")
        self.doc_rel = tf.placeholder(tf.float32, [None, self.doc_max_word,], "doc_rel")
        self.true_decision = tf.placeholder(tf.float32, [None,1,], "label")
        # learning rate
        self.lr = None
        self.sess = sess
        self.global_step = None

        self.logger = Logger()
        self.accuracy = dict()
        self.accuracy_op = dict()
        self.precision = dict()
        self.precision_op = dict()
        self.recall = dict()
        self.recall_op = dict()

        self.we = None


    def build_network(self):
        # we : word embedding
        we_mat = np.full((self.nwords,), 0)
        self.we = tf.Variable(initial_value=we_mat, dtype=tf.float32, trainable=True, name="word_embedding")

        doc_raw = tf.nn.embedding_lookup(self.we, self.doc_word)
        print_shape("doc_raw", doc_raw)
        print_shape("doc_rel", self.doc_rel)
        doc_contrv = tf.multiply(doc_raw, self.doc_rel)
        print_shape("doc_contrv", doc_contrv)

        # --- possibly sigmoid here ----
        decision = tf.reduce_sum(doc_contrv, 1)
        decision = tf.reshape(decision, [self.batch_size,1,])
        print_shape("decision", decision)
        return decision

    def binary_activate(self, decision):
        bias_value = -3.0
        bias = tf.constant(bias_value, dtype=tf.float32, shape=[self.batch_size, 1,])
        return tf.round(tf.sigmoid(tf.add(decision, bias)))

    def measure(self, y, y_true, run_names):
        # train / valid / test / etc
        for name in run_names:
            self.accuracy[name], self.accuracy_op[name] = tf.contrib.metrics.streaming_accuracy(y, y_true, name=name)
            self.precision[name], self.precision_op[name] = tf.contrib.metrics.streaming_precision(y, y_true, name=name)
            self.recall[name], self.recall_op[name] = tf.contrib.metrics.streaming_recall(y, y_true, name=name)

    def optimize(self, loss):
        self.lr = tf.Variable(self.current_lr, name="learning_rate")
        if self.optimizer == 'Adagrad':
            print("Using AdagradOptimizer")
            self.optim = tf.train.AdagradOptimizer(self.lr).minimize(loss)
        elif self.optimizer == 'Adam':
            print("Using AdamOptimizer")
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            print("Using GradientDescent")
            inc = self.global_step.assign_add(1)
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
            params = [self.we,]
            grads_and_vars = self.opt.compute_gradients(loss, params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]

            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)



    def build_model(self, run_names):
        print("Building Models")
        self.global_step = tf.Variable(0, name="global_step")

        decision = self.build_network()

        self.loss = tf.losses.mean_squared_error(decision, self.true_decision)

        y = self.binary_activate(decision)
        y_true = self.binary_activate(self.true_decision)

        self.measure(y, y_true, run_names)
        self.optimize(self.loss)



    def data2batch(self, data):
        def fill_up(list, length, value):
            new_list = list[:length]
            while len(new_list) < length:
                new_list.append(value)
            return new_list

        batch_supplier = Batch(self.batch_size)
        for test_case in data:
            (word_list, rel_list, label) = test_case
            word_list = fill_up(word_list, self.doc_max_word, 0)
            rel_list = fill_up(rel_list, self.doc_max_word, 0)

            a = np.ndarray(shape=(self.doc_max_word,), dtype=np.int32)
            for idx, item in enumerate(word_list):
                a[idx] = item
            b = np.ndarray(shape=(self.doc_max_word,), dtype=np.float32)
            for idx, item in enumerate(rel_list):
                b[idx] = item

            c = np.ndarray(shape=[1,], dtype=np.float32)
            c[0] = label
            single = (a, b, c)
            batch_supplier.enqueue(single)
        return batch_supplier

    def train(self, data):
        cost = 0
        n_test = len(data)
        debug = False

        batch_supplier = self.data2batch(data)
        debug = True
        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.doc_word: batch[0],
                self.doc_rel: batch[1],
                self.true_decision: batch[2],
            }
            (_, _, _, _,
            loss) = self.sess.run([self.optim, self.accuracy_op["train"], self.recall_op["train"], self.precision_op["train"],
                                               self.loss], feed_dict)

            cost += np.sum(loss)

            if debug :
                self.logger.print("Something", "content here")

        accuracy, precision, recall = self.sess.run(
            [self.accuracy["train"],
             self.precision["train"],
             self.recall["train"]])
        return cost/n_test, accuracy, precision, recall

    def test(self, data):
        n_test = len(data)
        cost = 0

        batch_supplier = self.data2batch(data)

        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.doc_word: batch[0],
                self.doc_rel: batch[1],
                self.true_decision: batch[2],
            }
            (loss, _, _, _,
             ) = self.sess.run([self.loss, self.accuracy_op["test"], self.precision_op["test"], self.recall_op["test"]
                                       ], feed_dict)
            cost += np.sum(loss)
        accuracy, precision, recall = self.sess.run(
            [self.accuracy["test"],
             self.precision["test"],
             self.recall["test"]])
        return cost/n_test, accuracy, precision, recall



    def run(self, train_data, test_data):
        print("Running...")
        if not self.is_test:
            train_acc_last = 0
            best_v_acc = 0
            for idx in range(self.nepoch):
                print("Epohc {}".format(idx))
                self.sess.run(tf.local_variables_initializer())
                tf.global_variables_initializer()
                print("I think i've done initilize")

                self.logger.set_prefix("Epoch:{}\n".format(idx))
                start = time.time()
                train_loss, train_acc, train_precision, train_recall = self.train(train_data)
                acc_delta = train_acc - train_acc_last
                train_acc_last = train_acc

                test_loss, test_acc, test_precision, test_recall = self.test(test_data)
                test_f  = 2*test_precision*test_recall/(test_precision+test_recall+0.001)

                elapsed = time.time() - start
                # Logging
                self.step, = self.sess.run([self.global_step])
                self.log_loss.append([train_loss, test_loss])

                state = {
                    'train_perplexity': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'train_accuracy': train_acc,
                    'acc_delta': acc_delta,
                    'valid_accuracy': test_acc,
                    'valid_f': test_f,
                    'elapsed': int(elapsed)
                }
                if self.print_state:
                    print(state)
                self.log.append(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx - 1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if test_acc > best_v_acc:
                    best_v_acc = test_acc
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "nn.model"),
                                    global_step=idx)
