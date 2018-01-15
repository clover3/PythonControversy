import numpy as np
import re
import itertools
from collections import Counter
from LRP.data_side import *

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def data_split(pos_path, neg_path):
    # 223 -> 74*3
    # 74*2 + alpha = 300
    #  = 300
    train_size = 300
    side_articles = load_side_articles()
    guardian_pos_text = list(open(pos_path, "r", encoding="utf-8").readlines())
    guardian_neg_text = list(open(neg_path, "r", encoding="utf-8").readlines())

    split_size = int(len(side_articles) / 3)

    side1 = side_articles[:split_size]
    side2 = side_articles[split_size:split_size*2]
    side3 = side_articles[split_size*2:]

    pos_splits = [((side1+side2),side3), ((side2+side3),side1), ((side3+side1),side2)]

    splits = []
    for pos_split, test_list in pos_splits:
        addition = train_size - len(pos_split)
        train_pos = extract_text(pos_split) + guardian_pos_text[:addition]
        train_neg = guardian_neg_text[:train_size]
        x_text = [clean_str(x) for x in train_pos + train_neg]
        y = [[0,1]] * len(train_pos) + [[1,0]] * len(train_neg)
        splits.append((x_text, np.array(y), test_list))

    return splits


# not used anymore
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
