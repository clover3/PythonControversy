from nltk.tokenize import wordpunct_tokenize
import collections
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import StratifiedKFold
import nltk

import pickle
import random
import math
import numpy as np
from clover_lib import *
import data_helpers

class Model:
    def __init__(self, x, y, split_no):
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
        self.good_phrase = data_helpers.load_phrase(split_no)
        self.split_no = split_no
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)


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
        pos_tags = nltk.pos_tag(text_tokens)
        def is_noun(idx):
            return 'NN' in pos_tags[idx][1]

        text_len = len(text_tokens)
        rand_range = text_len-phrase_len
        response = []
        for idx in range(rand_range):
            for gap in range(1, 3):
                if len(response) == k:
                    break
                idx = np.random.randint(0, rand_range)

                phrase = " ".join(text_tokens[idx : idx+gap])
                if phrase in self.good_phrase and is_noun(idx):
                    response.append(phrase)
        return response

    def gen_phrase(self, text_tokens, weight, phrase_len, k):
        pos_tags = nltk.pos_tag(text_tokens)
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

        def is_noun(idx):
            return 'NN' in pos_tags[idx][1]

        text_len = len(text_tokens)
        candidate = collections.Counter()
        for phrase_len in range(1,5):
            for idx in range(text_len-phrase_len):
                sub_tokens = text_tokens[idx : idx+phrase_len]
                last_idx = idx + phrase_len - 1

                phrase = " ".join(sub_tokens)
                if phrase in self.good_phrase and is_noun(last_idx):
                    score = sum(list([word_score(token) for token in sub_tokens]))
                    candidate[phrase] += score
        response = []
        for phrase, score in candidate.most_common():
            if phrase not in response:
                response.append(phrase)
            if len(response) == k:
                break
        return response

    @staticmethod
    def update_suc(suc, data, match):
        if match(data):
            suc.suc()
        else:
            suc.fail()

    def test_phrase(self, data, y, answers, k):
        tokens = [self.tokenize(s) for s in data]

        X = self.transform(tokens)
        y_pred = self.clf.predict(X)
        weights = np.multiply(X, self.clf.coef_[0])

        suc = FailCounter()
        rand_suc = FailCounter()
        responses= []
        responsesF = []
        for i, answer in enumerate(answers):
            if answer is None:
                continue
            if y_pred[i] != y[i]:
                #print("Wrong")
                ""

            answer_token = answer.lower().split(" ")

            answer_str = " ".join([self.stemmer.stem(t) for t in answer_token])

            candidates = self.gen_phrase(tokens[i], weights[i], len(answer_token), k)
            responses.append(candidates)
            rand_candi = self.rand_gen_phrase(tokens[i], len(answer_token), k)
            responsesF.append(rand_candi)
            print("answer ---: "+answer)
            for item in rand_candi:
                print(item)
            def match(tokens):
                systen_answer = " ".join([self.stemmer.stem(t) for t in tokens])
                return systen_answer == answer_str

            def correct(candidates):
               return any([match(c.split(" ")) for c in candidates])

            self.update_suc(rand_suc, rand_candi, correct)
            self.update_suc(suc, candidates, correct)

        pickle.dump(responses, open("middle.u.score{}.pickle".format(self.split_no), "wb"))
        pickle.dump(responsesF, open("middle.r.score{}.pickle".format(self.split_no), "wb"))
        print("accuracy : {}".format(suc.precision()))
        print("accuracy[First10] : {}".format(rand_suc.precision()))
        print("Total case : {}".format(suc.total()))
        return suc.precision(), rand_suc.precision()


def get(x, indice):
    res = []
    for index in indice:
        res.append(x[index])
    return res


if __name__ == "__main__":
    random.seed(0)

    # Load data
    print("Loading data...")
    validate = False
    splits = pickle.load(open("..\\splits.pickle", "rb"))
    for split_no, split in enumerate(splits):
        x_text, y, test_data = split
        y = np.argmax(np.array(y), axis=1)

        if not validate:
            svm_phrase = Model(x_text, y, split_no)
            svm_phrase.train(x_text, y)
            #print("Accuracy on train data: {}".format(svm_phrase.accuracy(x_text, y)))

            answers = data_helpers.load_answer(test_data)
            linkl, test_text = zip(*test_data)
            y_test = np.ones([len(test_text)])
            #print("Accuracy on contrv data: {}".format(svm_phrase.accuracy(test_text, y_test)))
            result = []
            rand_result = []
            for k in [10]:
                rate, rate_rand = svm_phrase.test_phrase(test_text, y_test, answers, k)
                result.append(rate)
                rand_result.append(rate_rand)

            print(result)
            print("First10 : ", end="")
            print(rand_result)

        else:
            print("---------------")
            skf = StratifiedKFold(n_splits=5)
            accuracys = []
            for train_index, test_index in skf.split(np.zeros(len(y)), y):
                x_train = get(x_text, train_index)
                x_test = get(x_text, test_index)
                y_train, y_test = y[train_index], y[test_index]
                print("total {}, pos={}".format(len(y_test), np.count_nonzero(y_test)))
                svm_phrase = Model(x_train, y_train, split_no)
                svm_phrase.train(x_train, y_train)
                print("Accuracy on test data: {}".format(svm_phrase.accuracy(x_test, y_test)))
            break
        print("--- Done split-----")
