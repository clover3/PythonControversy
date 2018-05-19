import collections
import math
import pickle
import random

import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import data_helpers
from clover_lib import *


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

        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.good_phrase = data_helpers.load_phrase(split_no)
        self.split_no = split_no

    def tokenize(self, str):
        return wordpunct_tokenize(str)

    def log_odd(self, tokens):
        smoothing = 0.1
        def count(LM, token):
            if token in LM:
                return LM[token]
            else:
                return 0

        def per_token_odd(token):
            tf_c = count(self.C, token)
            tf_nc = count(self.NC, token)
            if tf_c == 0 and tf_nc == 0 :
                return 0
            P_w_C = tf_c / self.C_ctf
            P_w_NC = tf_nc / self.NC_ctf
            P_w_BG = (tf_c+tf_nc) / (self.NC_ctf + self.C_ctf)
            logC = math.log(P_w_C * smoothing + P_w_BG * (1-smoothing))
            logNC = math.log(P_w_NC * smoothing + P_w_BG * (1 - smoothing))
            assert(math.isnan(logC)==False)
            assert (math.isnan(logNC) == False)
            return logC - logNC

        sum_odd = 0
        for token in tokens:
            sum_odd += per_token_odd(token)
        return sum_odd

    def train(self, x, y):
        self.NC = collections.Counter()
        self.C = collections.Counter()
        def update(counter, tokens):
            for token in tokens:
                counter[token] += 1

        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            if y[idx] == 0:
                update(self.NC, tokens)
            elif y[idx] == 1:
                update(self.C, tokens)

        self.NC_ctf = sum(self.NC.values())
        self.C_ctf = sum(self.C.values())

        vectors = []
        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            odd = self.log_odd(tokens)
            vectors.append((odd, y[idx]))
        vectors.sort(key=lambda x:x[0], reverse=True)

        total = len(vectors)
        p =  np.count_nonzero(y)
        fp = 0

        max_acc = 0
        self.opt_alpha = 0
        for idx, (odd, label) in enumerate(vectors):
            alpha = odd - 1e-8
            if label == 0:
                fp += 1

            tp = (idx+1) - fp
            fn = p - tp
            tn = total - (idx+1) - fn
            acc = (tp + tn) / (total)
            if acc > max_acc:
                self.opt_alpha = alpha
                max_acc = acc

        print("Train acc : {}".format(max_acc))

    def predict(self, data):
        y = []
        for idx, s in enumerate(data):
            tokens = self.tokenize(s)
            odd = self.log_odd(tokens)
            y.append(int(odd > self.opt_alpha))

        return np.array(y)

    def accuracy(self, x, y):
        return accuracy_score(y, self.predict(x))

    def gen_phrase(self, text, k):
        text_tokens = self.tokenize(text)
        pos_tags = nltk.pos_tag(text_tokens)

        per_word_score = collections.Counter()
        for token in text_tokens:
            per_word_score[token] += self.log_odd([token])
        avg_word_score = sum(per_word_score.values())/ len(per_word_score)

        def word_score(word):
            if word in per_word_score:
                return per_word_score[word]
            else:
                return avg_word_score

        def is_noun(idx):
            return 'NN' in pos_tags[idx][1]


        candidate = collections.Counter()
        text_len = len(text_tokens)
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
        y_pred = self.predict(data)
        suc = FailCounter()

        responses= []
        for i, answer in enumerate(answers):

            answer_token = answer.lower().split(" ")

            answer_str = " ".join([self.stemmer.stem(t) for t in answer_token])
            candidates = self.gen_phrase(data[i], k)
            responses.append(candidates)
            def match(tokens):
                systen_answer = " ".join([self.stemmer.stem(t) for t in tokens])
                return systen_answer == answer_str

            def correct(candidates):
               return any([match(c.split(" ")) for c in candidates])
            self.update_suc(suc, candidates, correct)

        pickle.dump(responses, open("middle.lm.score{}.pickle".format(self.split_no), "wb"))
        print("accuracy : {}".format(suc.precision()))
        print("Total case : {}".format(suc.total()))
        return suc.precision()

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
            LM = Model(x_text, y, split_no)
            LM.train(x_text, y)
            #print("Accuracy on train data: {}".format(svm_phrase.accuracy(x_text, y)))

            answers = data_helpers.load_answer(test_data)
            linkl, test_text = zip(*test_data)
            y_test = np.ones([len(test_text)])
            print("Accuracy on contrv data: {}".format(LM.accuracy(test_text, y_test)))
            result = []
            for k in [1]:
                rate = LM.test_phrase(test_text, y_test, answers, k)
                result.append(rate)

            print(result)
        else:
            print("---------------")
            skf = StratifiedKFold(n_splits=5)
            accuracys = []
            for train_index, test_index in skf.split(np.zeros(len(y)), y):
                x_train = get(x_text, train_index)
                x_test = get(x_text, test_index)
                y_train, y_test = y[train_index], y[test_index]
                print("total {}, pos={}".format(len(y_test), np.count_nonzero(y_test)))
                LM = Model(x_train, y_train, split_no)
                LM.train(x_train, y_train)
                print("Accuracy on test data: {}".format(LM.accuracy(x_test, y_test)))
            break
        print("--- Done split-----")
