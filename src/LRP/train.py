#! /usr/bin/env python

import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import os
import sys
import data_helpers
from LRP.cnn import TextCNN
from LRP.lrp import LRPManager
from LRP.manager import Manager
from sklearn.model_selection import StratifiedKFold

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
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
    splits = data_helpers.data_split(pos_path, neg_path)
    x_text, y, test_data = splits[0]

    # Build vocabulary
    max_document_length = 5000
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    skf = StratifiedKFold(n_splits=5)
    accuracys = []
    for train_index, test_index in skf.split(x, np.argmax(y, axis=1)):
        x_train, x_dev = x[train_index], x[test_index]
        y_train, y_dev = y[train_index], y[test_index]

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        initW = data_helpers.load_word2vec(vocab_processor, FLAGS.embedding_dim)

        # Training
        # ==================================================
        with tf.device('/device:GPU:1'), tf.Graph().as_default():
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
                todo = "train"
                if todo == "lrp_test":
                    manager.show_lrp(sess, lrp, x_dev, y_dev, x_dev_text)
                    #manager.word_removing(sess, lrp, x_dev[:30], y_dev[:30])
                elif todo == "phr_test":
                    # test_data : list(link, text)
                    answer = data_helpers.load_answer(test_data)
                    manager.test_phrase(sess, lrp, test_data, answer, vocab_processor)
                elif todo == "heatmap":
                    answer = data_helpers.load_answer(test_data)
                    manager.heatmap(sess, lrp, test_data, answer, vocab_processor)
                elif todo == "train2phrase":
                    answer = data_helpers.load_answer(test_data)
                    manager.train_and_phrase(sess, lrp, test_data, answer, vocab_processor)
                else:
                    accuracy = manager.train(sess, cnn, x_train, x_dev, y_train, y_dev)
                    accuracys.append(accuracy)

    print(accuracys)