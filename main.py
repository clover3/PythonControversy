import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from guardian_data import load_data
import tensorflow as tf
import random
from model_common import *
from model import *
from view import *
import pickle
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 10, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 200, "number of epoch to use during training [100]")
flags.DEFINE_integer("doc_max_word", 150, "max number of word in a doc")

flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_string("optimizer", "Adam", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", True, "print progress [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_dir", "checkpoint_dir")

if "__main__" == __name__ :
    random.seed(0)
    corpus_path = os.path.join("data", "RelevanceWord.txt")
    label_path = os.path.join("data", "Label.txt")
    data, voca_size, word2idx = load_data(corpus_path, label_path)
    flags.DEFINE_integer("nwords", voca_size, "number of words in corpus")
    runs = split_train_test(data, 3)

    summary = []
    we_arr = []
    for (train_data, valid_data) in runs:
        with tf.Session() as sess:
            model = ContrvWord(flags.FLAGS, sess)
            model.build_model(run_names=["train", "test"])
            tf.global_variables_initializer().run()
            if flags.FLAGS.is_test:
                print("Test not implemented")
            else: #train
                model.run(train_data, valid_data)
                best_accuracy = model.get_best_valid('valid_accuracy')
                best_f = model.get_best_valid('valid_f')
                summary.append("best_accuract : {}\t best_f : {}".format(best_accuracy, best_f))
            we_arr.append(model.out_we)
    for s in summary:
        print(s)
    pickle.dump(we_arr, open("we_arr.pickle", "wb"))
    print_word_embedding(word2idx, we_arr)

    play_process_completed()
