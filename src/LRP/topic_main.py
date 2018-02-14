#! /usr/bin/env python

import pickle
import random

import numpy as np
import tensorflow as tf
import os
import data_helpers
from topic_manager import TopicManager
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_classes", 2, "Number of classes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_topics", 3000, "Number of topics (default: 3000)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("topic_lambda", 10, "topic regularization lambda (default: 0.0)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 42, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 48, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
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
    todo = "phrase_test"
    # Load data
    print("Loading data...")
    #splits = data_helpers.data_split(pos_path, neg_path)
    #pickle.dump(splits, open("splits.pickle", "wb"))
    splits = pickle.load(open("splits.pickle", "rb"))
    for split_no, split in enumerate(splits):
        print("Split {}".format(split_no))
        x_text, y, test_data = splits[split_no]

        vocab_processor = pickle.load(open("vocabproc{}.pickle".format(split_no), "rb"))
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                manager = TopicManager(num_checkpoints=FLAGS.num_checkpoints,
                                  dropout_keep_prob=FLAGS.dropout_keep_prob,
                                  batch_size=FLAGS.batch_size,
                                  num_epochs=FLAGS.num_epochs,
                                  evaluate_every=FLAGS.evaluate_every,
                                  checkpoint_every=FLAGS.checkpoint_every,
                                  )
                if todo == "phrase_test":
                    # test_data : list(link, text)
                    answer = data_helpers.load_answer(test_data)
                    manager.test_phrase(sess, FLAGS, vocab_processor, test_data, answer, split_no, k=10)
                elif todo == "train":
                    manager.train(sess, x_text, y, split_no, FLAGS)
                else:
                    print("Unknown command")
        break