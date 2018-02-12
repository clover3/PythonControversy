import collections
import numpy as np
import re
import itertools
from collections import Counter
from LRP.data_side import *
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import StratifiedKFold
import os
from random import shuffle
from clover_lib import *


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


def load_word2vec(vocab_processor, embedding_dim):
    path = os.path.join("data", "word2vec")
    model = Word2Vec.load(path)

    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), embedding_dim))
    total = len(vocab_processor.vocabulary_._mapping.keys())
    cnt = 0
    for word in vocab_processor.vocabulary_._mapping.keys():
        if word in model.wv:
            v = model.wv[word]
            assert (v.size == embedding_dim)
            cnt = cnt+ 1
            idx = vocab_processor.vocabulary_.get(word)
            initW[idx] = v
    #print("voca : {}/{} initialized".format(cnt, total))
    return initW


def valid_side_articles():
    path = "C:\work\Code\Isidewith\\article_plain.pickle"
    plain_articles = pickle.load(open(path, "rb"))
    # dict of key -> content
    # content contains newline(\n)

    topic_convert = {
        'Trident Nuclear Weapons Programme':'Nuclear Weapon',
        'Deportation of Suspected Terrorists':'Suspected Terrorists',
    }

    topic_list = ["European Union",
        "Nuclear Weapon",
        "Foreign Aid",
        "Mental Health",
        "Tuition Fees",
        "Immigration",
        "Abortion",
        "Death Penalty",
        "Taxes",
        "Welfare",
        "Free Trade",
        "Bitcoin",
        "House of Lords"]


    # link -> article
    # if not in dict, not valid
    answer_dict = dict()
    issue_path = "C:\work\Data\isidewith\pickle\issue_with_article.pickle"
    issues = pickle.load(open(issue_path , "rb"))
    for issue in issues:
        topic_title = issue['title']
        if topic_title in topic_convert:
            topic_title = topic_convert[topic_title]
        articles = issue['articles']
        if topic_title not in topic_list:
            continue

        for title, link in articles:
            if link.startswith("/poll/"):
                continue
            try:
                text = plain_articles[link]
                if topic_title.lower() in text.lower():
                    answer_dict[link] = topic_title
            except KeyError:
                "Means not crawled"

    topicCount = collections.Counter()
    data = []
    for (link, text) in plain_articles.items():
        if link in answer_dict:
            topicCount[answer_dict[link]] += 1
            data.append((link, text))
    print("{} valid side article ".format(len(data)))
    for topic in topicCount:
        print("{} : {}".format(topic, topicCount[topic]))
    return data


def data_count():

    issue_path = "C:\work\Data\isidewith\pickle\issue_with_article.pickle"
    issues = pickle.load(open(issue_path , "rb"))
    aritlce_path = "C:\work\Data\isidewith\pickle\\article_plain.pickle"
    plain_articles = pickle.load(open(aritlce_path, "rb"))


    pos_path = "data\\guardianC.txt"
    neg_path = "data\\guardianNC.txt"
    side_articles = load_side_articles()
    print("At first {}".format(len(side_articles)))

    topic_convert = {
        'Trident Nuclear Weapons Programme':'Nuclear Weapon',
        'Deportation of Suspected Terrorists':'Suspected Terrorists',
    }

    answer_dict = dict()
    for issue in issues:
        topic_title = issue['title']
        if topic_title in topic_convert:
            topic_title = topic_convert[topic_title]

    status = dict()
    answer_dict = dict()
    for issue in issues:
        topic_title = issue['title']
        if topic_title in topic_convert:
            topic_title = topic_convert[topic_title]
        articles = issue['articles']
        if len(articles) <= 4:
            for title, link in articles:
                status[link] = "few :" + topic_title
            continue

        for title, link in articles:
            if link.startswith("/poll/"):
                continue
            try:
                text = plain_articles[link]
                if topic_title.lower() in text.lower():
                    answer_dict[link] = topic_title
                    status[link] = topic_title
                else:
                    status[link] = "no answer : " + topic_title
            except KeyError:
                ""


    answer = []
    count =0
    for link, text in side_articles:
        print(status[link])
        if link in answer_dict:
            answer.append(answer_dict[link])
        else:
            count = count + 1
            answer.append(None)
    print("{} of {} not found ".format(count, len(side_articles)))
    return answer


def data_split(pos_path, neg_path):
    # 223 -> 74*3
    # 74*2 + alpha = 300
    #  = 300
    train_size = 300
    side_articles = valid_side_articles()
    shuffle(side_articles)
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
        splits.append((x_text, y, test_list))

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


def load_phrase(split_no):
    path = "C:\work\Code\PythonControversy\src\LRP\\" +"phrase{}.pickle".format(split_no)
    l_candidate_phrase = pickle.load(open(path,"rb"))
    candidate_phrase = []
    for p in l_candidate_phrase:
        candidate_phrase.append(p)
        #candidate_phrase.append(" ".join(p))
    print("{} candidate phrase loaded".format(len(candidate_phrase)))
    return candidate_phrase


def load_answer(test_data):
    issue_path = "C:\work\Data\isidewith\pickle\issue_with_article.pickle"
    issues = pickle.load(open(issue_path , "rb"))
    aritlce_path = "C:\work\Data\isidewith\pickle\\article_plain.pickle"
    plain_articles = pickle.load(open(aritlce_path, "rb"))
    topic_convert = {
        'Trident Nuclear Weapons Programme':'Nuclear Weapon',
        'Deportation of Suspected Terrorists':'Suspected Terrorists',
    }

    answer_dict = dict()
    for issue in issues:
        topic_title = issue['title']
        if topic_title in topic_convert:
            topic_title = topic_convert[topic_title]

        articles = issue['articles']
        if len(articles) <= 4:
            continue

        for title, link in articles:
            if link.startswith("/poll/"):
                continue
            try:
                text = plain_articles[link]
                if topic_title.lower() in text.lower():
                    answer_dict[link] = topic_title
            except KeyError:
                ""
    answer = []

    count =0
    for link, text in test_data:
        if link in answer_dict:
            answer.append(answer_dict[link])
        else:
            answer.append(None)
    #print("{} of {} not found ".format(count, len(test_data)))
    return answer



def get_fold(n):
    splits = pickle.load(open("splits.pickle", "rb"))
    x_text, y, test_data = splits[0]
    skf = StratifiedKFold(n_splits=n)
    splits = []
    for train_index, test_index in skf.split(x_text, y):
        print("TRAIN:", train_index, "TEST:", test_index)


def check_goodphrase():
    splits = pickle.load(open("LRP\\splits.pickle", "rb"))

    count = collections.Counter()
    for split_no, split in enumerate(splits):
        phrases = load_phrase(split_no)
        x_text, y, test_data = split
        answers = load_answer(test_data)
        for answer in answers:
            count[len(answer.split(" "))] += 1

    for key in count:
        print("{} : {}".format(key, count[key]))


