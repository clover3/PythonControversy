from nltk.tokenize import wordpunct_tokenize
import collections
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import random
import math
import numpy as np
from clover_lib import *
import data_helpers

class Model:
    def __init__(self, x, y):
        tokened_corpus = [self.tokenize(s) for s in x]

        voca = collections.Counter()
        for tokens in tokened_corpus:
            for token in tokens:
                voca[token] += 1
        self.N = len(tokened_corpus)
        self.active_voca = []
        for word in voca.keys():
            if voca[word] > 5:
                self.active_voca.append(word)

        self.df= collections.Counter()
        for tokens in tokened_corpus:
            for token in set(tokens):
                self.df[token] += 1

        self.word2idx = dict()
        for idx, word in enumerate(self.active_voca):
            self.word2idx[word] = idx
        self.dim = len(self.word2idx)
        self.clf = None

    def tokenize(self, str):
        return wordpunct_tokenize(str)

    @staticmethod
    def count(tokens):
        voca = collections.Counter()
        for token in tokens:
            voca[token] += 1
        return voca

    def transform(self, text_list):
        result = []
        for tokens in text_list:
            voca = self.count(tokens)

            vector = np.zeros([self.dim])
            for word in voca.keys():
                if word in self.word2idx:
                    tf = voca[word]
                    idf = math.log(self.N/self.df[word])
                    vector[self.word2idx[word]] = tf * idf
            result.append(vector)
        return np.array(result)

    def train(self, x, y):
        tokens = [self.tokenize(s) for s in (x)]

        X = self.transform(tokens)

        self.clf = LinearSVC()
        self.clf.fit(X, y)

    def predict(self, data):
        tokens = [self.tokenize(s) for s in (data)]
        X = self.transform(tokens)
        return self.clf.predict(X)

    def accuracy(self, x, y):
        return accuracy_score(y, self.predict(x))

    def rand_gen_phrase(self, text_tokens, phrase_len, k):
        text_len = len(text_tokens)
        rand_range = min(text_len-phrase_len, 500)
        for idx in np.random.randint(0, rand_range, k):
            yield " ".join(text_tokens[idx : idx+phrase_len])

    def gen_phrase(self, text_tokens, weight, phrase_len, k):
        voca = self.count(text_tokens)
        per_word_score = dict()
        for word in voca.keys():
            if word in self.word2idx:
                per_word_score[word] = weight[self.word2idx[word]] / voca[word]
        avg_word_score = sum(per_word_score.values())/ len(per_word_score)

        def word_score(word):
            if word in per_word_score:
                return per_word_score[word]
            else:
                return avg_word_score

        text_len = len(text_tokens)
        candidate = []
        for idx in range(text_len-phrase_len):
            sub_tokens = text_tokens[idx : idx+phrase_len]
            score = sum(list([word_score(token) for token in sub_tokens]))
            candidate.append((score, text_tokens[idx:idx+phrase_len]))
        candidate.sort(key=lambda x:x[0], reverse=True)
        response = set()
        for score, tokens in candidate:
            phrase = " ".join(tokens)
            if phrase not in response:
                response.add(phrase)
            if len(response) == k:
                break
        return response

    @staticmethod
    def update_suc(suc, data, match):
        if match(data):
            suc.suc()
        else:
            suc.fail()

    def test_phrase(self, data, y, answers):
        tokens = [self.tokenize(s) for s in data]
        X = self.transform(tokens)
        y_pred = self.clf.predict(X)
        weights = np.multiply(X, self.clf.coef_[0])

        suc = FailCounter()
        rand_suc = FailCounter()
        for i, answer in enumerate(answers):
            if y_pred[i] != y[i]:
                continue
            if answer is None:
                continue
            answer_token = answer.lower().split(" ")

            candidates = self.gen_phrase(tokens[i], weights[i], len(answer_token), 10)
            rand_candi = self.rand_gen_phrase(tokens[i], len(answer_token), 10)

            def match(tokens):
                return set(tokens) == set(answer_token)

            def correct(candidates):
               return any([match(c.split(" ")) for c in candidates])

            self.update_suc(suc, candidates, correct)
            self.update_suc(rand_suc, rand_candi, correct)

        print("accuracy : {}".format(suc.precision()))
        print("accuracy[random] : {}".format(rand_suc.precision()))
        print("Total case : {}".format(suc.total()))


if __name__ == "__main__":
    random.seed(0)

    pos_path = "..\\data\\guardianC.txt"
    neg_path = "..\\data\\guardianNC.txt"
    # Load data
    print("Loading data...")
    validate = True
    splits = data_helpers.data_split(pos_path, neg_path)
    for split in splits:
        x_text, y, test_data = split
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_text = list(np.array(x_text)[shuffle_indices])
        y = y[shuffle_indices]
        y = np.argmax(y, axis=1)

        if not validate:
            svm_phrase = Model(x_text, y)
            svm_phrase.train(x_text, y)
            print("Accuracy on train data: {}".format(svm_phrase.accuracy(x_text, y)))

            answers = data_helpers.load_answer(test_data)
            linkl, test_text = zip(*test_data)
            print("Accuracy on contrv data: {}".format(svm_phrase.accuracy(test_text, np.ones([len(test_text)]))))
            svm_phrase.test_phrase(test_text, y, answers)

        else:
            hold_out = 0.2
            split_idx = int(len(x_text) * (1- hold_out))

            train_x = x_text[:split_idx]
            train_y = y[:split_idx]
            dev_x = x_text[split_idx:]
            dev_y = y[split_idx:]

            svm_phrase = Model(x_text, y)
            svm_phrase.train(train_x, train_y)
            print("Accuracy on train data: {}".format(svm_phrase.accuracy(train_x, train_y)))
            print("Accuracy on dev data: {}".format(svm_phrase.accuracy(dev_x, dev_y)))

            answers = data_helpers.load_answer(test_data)
            linkl, test_text = zip(*test_data)
            print("Accuracy on contrv data: {}".format(svm_phrase.accuracy(test_text, np.ones([len(test_text)])) ))
            svm_phrase.test_phrase(test_text, y, answers)
