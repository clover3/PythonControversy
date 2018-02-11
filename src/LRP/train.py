#! /usr/bin/env python

import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import os
import data_helpers
from LRP.cnn import TextCNN
from LRP.lrp import LRPManager
from LRP.manager import Manager

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
if "__main__" == __name__ :
    random.seed(0)


    # Data Preparation
    # ==================================================
    pos_path = os.path.join("data","guardianC.txt")
    neg_path = os.path.join("data","guardianNC.txt")

    # Load data
    print("Loading data...")
    #splits = data_helpers.data_split(pos_path, neg_path)
    #pickle.dump(splits, open("splits.pickle", "wb"))
    splits = pickle.load(open("splits.pickle", "rb"))
    for split_no, split in enumerate(splits):
        print("Split {}".format(split_no))
        x_text, y, test_data = splits[split_no]

        # Build vocabulary
        max_document_length = 5000
        #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        vocab_processor = pickle.load(open("vocabproc{}.pickle".format(split_no), "rb"))
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        y = np.array(y)

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        text_x_shuffled = []
        for index in np.nditer(shuffle_indices):
            text_x_shuffled.append(x_text[index])


        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        x_dev_text = text_x_shuffled[dev_sample_index:]
        pickle.dump(x_dev_text, open("dev_x_text.pickle","wb"))
        del x, y, x_shuffled, y_shuffled

        #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        #print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        initW = data_helpers.load_word2vec(vocab_processor, FLAGS.embedding_dim)

        # Training
        # ==================================================

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    batch_size=FLAGS.batch_size,
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
                lrp = LRPManager(sess,
                                 cnn=cnn,
                                 sequence_length=x_train.shape[1],
                                 num_classes=y_train.shape[1],
                                 vocab_size=len(vocab_processor.vocabulary_),
                                 batch_size=FLAGS.batch_size,
                                 embedding_size=FLAGS.embedding_dim,
                                 filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                                 num_filters=FLAGS.num_filters,
                                 )

                manager = Manager(num_checkpoints=FLAGS.num_checkpoints,
                                  dropout_keep_prob=FLAGS.dropout_keep_prob,
                                  batch_size=FLAGS.batch_size,
                                  num_epochs=FLAGS.num_epochs,
                                  evaluate_every=FLAGS.evaluate_every,
                                  checkpoint_every=FLAGS.checkpoint_every,
                                  initW=initW
                                  )
                todo = "demo_sentence"
                if todo == "lrp_test":
                    manager.show_lrp(sess, lrp, x_dev, y_dev, x_dev_text, split_no)
                    #manager.word_removing(sess, lrp, x_dev[:30], y_dev[:30])
                elif todo == "phrase_test":
                    # test_data : list(link, text)
                    answer = data_helpers.load_answer(test_data)
                    manager.test_phrase(sess, lrp, test_data, answer, vocab_processor, split_no, 1)
                elif todo== "test1510":
                    answer = data_helpers.load_answer(test_data)
                    pickle.dump(answer, open("answer{}.pickle".format(split_no), "wb"))
                    summary = []
                    for k in [10,]:
                        rate = manager.test_phrase(sess, lrp, test_data, answer, vocab_processor, split_no, k)
                        summary.append("{}\t{}".format(k, rate))
                    for line in summary:
                        print(line)
                elif todo == "heatmap":
                    answer = data_helpers.load_answer(test_data)
                    manager.heatmap(sess, lrp, test_data, answer, vocab_processor)
                elif todo == "train2phrase":
                    answer = data_helpers.load_answer(test_data)
                    manager.train_and_phrase(sess, lrp, test_data, answer, vocab_processor)
                elif todo == "demo_sentence":
                    answer = data_helpers.load_answer(test_data)
                    manager.sentence(sess, lrp, test_data, answer, vocab_processor, split_no)
                else:
                    manager.train(sess, cnn, x_train, x_dev, y_train, y_dev)
