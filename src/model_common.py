
from clover_lib import *
import math
import random
import numpy as np
import tensorflow as tf
from itertools import groupby

def split_train_test(all_data, n_fold=3):
    random.shuffle(all_data)

    size = len(all_data)
    fold_size = int(size / n_fold)
    folds = []
    for i in range(n_fold):
        folds.append(fold_size*i)

    r = []
    f = open("test_split.txt","w")
    for test_idx in range(n_fold):
        mid1 = test_idx*fold_size
        mid2 = (test_idx+1)*fold_size
        train = all_data[0:mid1] + all_data[mid2:]
        test  = all_data[mid1:mid2]
        print("test_idx = {} train len = {} test len = {}".format(test_idx, len(train), len(test)))
        r.append((train, test))
        for entry in test:
            article_id = entry[3]
            f.write(article_id + "\t")
        f.write("\n")
    f.close()
    return r


def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)


def to_unit_vector(tensor):
    sum = tf.reduce_sum(tensor, 1)
    return tensor / sum


def assert_shape(tensor, shape):
    if tensor.shape != shape:
        print_shape("Tensor shape error - ", tensor)
    assert (tensor.shape == shape)


def cap_by_one(tensor, shape):
    ones = tf.ones(shape)
    tensor = tf.minimum(tensor, ones)
    return tensor


def probability_and(tensor, axis):
    return tf.reduce_sum(tensor, axis)
    one = tf.ones_like(tensor)
    cap = tf.minimum(tensor, one)
    nots = one - cap

    not_prob = tf.reduce_prod(nots, axis)
    ones = tf.ones_like(not_prob)
    return ones - not_prob


def activate_label(prob_tensor):
    bias_value = -0.5
    bias = tf.constant(bias_value, dtype=tf.float32, shape=prob_tensor.shape)
    return tf.round(tf.sigmoid(tf.add(prob_tensor, bias)))

