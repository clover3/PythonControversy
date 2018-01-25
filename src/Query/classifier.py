from nltk.tokenize import wordpunct_tokenize
import collections
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import pickle
import time
import random
import math
import numpy as np
from clover_lib import *
import data_helpers

class BM25_dict():
    def __init__(self):
        pickle_path = "bm25.pickle"
        #self.database = pickle.load(open(pickle_path,"rb"))
        self.database = dict()

    @staticmethod
    def get_keystring(doc_sparse):
        key_string = ""
        for key in doc_sparse.keys()[10:]:
            key_string += "{}{}".format(key, doc_sparse[key])
        return key_string

    def bm25(self, doc_sparse):
        keystr = self.get_keystring(doc_sparse)
        if keystr in self.database:
            return self.database[keystr]
        else:
            return None

    def update(self, doc_sparse, history):
        keystr = self.get_keystring(doc_sparse)
        if keystr in self.database:
            db = self.database[keystr]
            for key in history:
                db[key] = history[key]
            self.database[keystr] = db
        else:
            self.database[doc_sparse] = history

def load_stopwords():
    s = set()
    f = open("..\\..\\data\\stopwords.dat", "r")
    for line in f.readline():
        s.add(line.strip())
    return s

class Model:
    def __init__(self):
        self.k1 = 1.2
        self.k2 = 10
        self.b = 0.75
        self.avgDocLen = 1000
        self.bm25_dict = BM25_dict()
        self.phrases= None # init at train

    # phrase : list[word]
    def bm25(self, doc_sparse, phrase):
        doclen = sum(doc_sparse.values())
        K = self.k1 * ((1 - self.b) + self.b * doclen / self.avgDocLen)

        def get_tf(word):
            if word in doc_sparse:
                return doc_sparse[word]
            else:
                return 0

        score = 0
        for word in phrase:
            idf = (self.n_docs - self.df[word]+0.5)/(self.df[word]+0.5)
            tf = get_tf(word)
            doc_side = tf * (1+self.k1) / (tf + self.k1 * K)
            query_side = 1 * (1+self.k2) / (1 + self.k2)
            score = score + math.log(idf * doc_side * query_side + 0.0001)
        return score


    def tokenize(self, str):
        return wordpunct_tokenize(str)

    def train(self, x , y):
        self.tokened_corpus = [self.tokenize(s) for s in x]
        self.n_docs = len(self.tokened_corpus)

        self.df = collections.Counter()
        voca = collections.Counter()
        self.doc_sparse = []
        for tokens in self.tokened_corpus:
            doc_sparse = collections.Counter()
            for token in tokens:
                voca[token] += 1
                doc_sparse[token] += 1
            for key in doc_sparse:
                self.df[key] += 1
            self.doc_sparse.append(doc_sparse)
        self.total_words = sum(voca.values())

        candidate_phrase = self.generate_feature_phrases(self.tokened_corpus, voca, self.total_words)
        self.phrases = list([p.split(" ") for p in candidate_phrase])
        pickle.dump(self.phrases, open("phrase.pickle","wb"))

        start = time.time()
        #X = pickle.load(open("trainX.pickle", "rb"))
        X = self.transform(self.tokened_corpus)
        pickle.dump(X, open("trainX.pickle", "wb"))
        elapsed = time.time() - start
        print("BM25 takes {}".format(elapsed))
        X = np.array(X)
        self.clf = LinearSVC()
        self.clf.fit(X, y)


    def generate_feature_phrases(self, tokened_corpus, voca, total_words):
        n_docs = len(tokened_corpus)
        stopwords = load_stopwords()

        def all_stopword(phrase):
            return all([token in stopwords for token in phrase])

        phrase_doc_rep = []
        for doc in tokened_corpus:
            phrase_list = collections.Counter()
            for phrase_len in [1, 2, 3]:
                for i in range(0, len(doc)-phrase_len):
                    phrase = doc[i:i+phrase_len]
                    phrase_list[" ".join(phrase)] += 1
            phrase_doc_rep.append(phrase_list)

        phrase_df = collections.Counter()
        phrase_f = collections.Counter()
        for phrase_list in phrase_doc_rep:
            for phrase in phrase_list.keys():
                phrase_df[phrase] += 1
                phrase_f[phrase] += phrase_list[phrase]

        print("All phrase {}".format(len(phrase_df)))

        def meaningful(phrase):
            token_phrase = phrase.split(" ")
            if all_stopword(phrase):
                return False
            for i, word in enumerate(token_phrase):
                try:
                    if i==0:
                        continue
                    prefix = " ".join(token_phrase[:i])
                    cur_phrase = " ".join(token_phrase[:i+1])
                    tf = voca[word]
                    p_b = tf / total_words
                    p_b_bar_a = phrase_f[cur_phrase] / phrase_f[prefix]
                    if p_b_bar_a < p_b * 2:
                        return False
                except ZeroDivisionError:
                    raise

            return True


        def very_meaningful(phrase):
            token_phrase = phrase.split(" ")
            if all_stopword(phrase):
                return False
            for i, word in enumerate(token_phrase):
                try:
                    if i==0:
                        continue
                    prefix = " ".join(token_phrase[:i])
                    cur_phrase = " ".join(token_phrase[:i+1])
                    tf = voca[word]
                    p_b = tf / total_words
                    p_b_bar_a = phrase_f[cur_phrase] / phrase_f[prefix]
                    if p_b_bar_a < p_b * 10:
                        return False
                except ZeroDivisionError:
                    raise

            return True

        candidate_phrase = []
        for phrase in phrase_f.keys():
            if phrase_df[phrase]> 3 and meaningful(phrase):
                candidate_phrase.append(phrase)
        print("filtered phrase {}".format(len(candidate_phrase)))
        return set(candidate_phrase)

    def accuracy(self, x, y):
        return accuracy_score(y, self.predict(x))

    @staticmethod
    def doc2sparse(tokens):
        sparse = collections.Counter()
        for t in tokens:
            sparse[t] += 1
        return sparse

    def transform(self, docs):
        X = []
        for doc in docs:
            max_idx = 0
            max_val =0
            doc = self.doc2sparse(doc)
            voca_here = set(doc.keys())
            x = []
            #history = dict()
            countMatch =  0
            for i, phrase in enumerate(self.phrases):
                if any([word in voca_here for word in phrase]):
                    rel = self.bm25(doc, phrase)
                else:
                    rel = 0
                if rel > max_val:
                    max_val = rel
                    max_idx = i
                if rel < 0 :
                    rel = 0
                x.append(rel)
                #history[phrase] = rel
            X.append(np.array(x))
        X = np.array(X)
        return X

    def predict(self, data):
        docs = [self.tokenize(s) for s in (data)]
        X = self.transform(docs)
        return self.clf.predict(X)

    @staticmethod
    def update_suc(suc, data, match):
        if match(data):
            suc.suc()
        else:
            suc.fail()

    def gen_phrase(self, X, weight, phrase_len, k):
        candidate = []
        for i, phrase in enumerate(self.phrases):
            score = weight[i]
            if len(phrase) == phrase_len:
                candidate.append((score, X[i], phrase))
        candidate.sort(key=lambda x: x[0], reverse=True)
        response = set()
        for score, bm25, tokens in candidate:
            phrase = " ".join(tokens)
            if phrase not in response:
                response.add(phrase)
            if len(response) == k:
                break
        return response

    def test_phrase(self, data, y, answers):
        tokens = [self.tokenize(s) for s in data]
        #X = pickle.load(open("testX.pickle", "rb"))
        X = self.transform(tokens)
        pickle.dump(X, open("testX.pickle", "wb"))
        y_pred = self.clf.predict(X)
        max_topic = np.argmax(X,axis=1)
        weights = np.multiply(X, self.clf.coef_[0])

        suc = FailCounter()
        for i, answer in enumerate(answers):

            if y_pred[i] != y[i]:
                continue
            if answer is None:
                continue
            answer_token = answer.lower().split(" ")
            j = max_topic[i]
            candidates = self.gen_phrase(X[i], weights[i], len(answer_token), 10)

            def match(tokens):
                return set(tokens) == set(answer_token)

            def correct(candidates):
               return any([match(c.split(" ")) for c in candidates])

            self.update_suc(suc, candidates, correct)

        print("accuracy : {}".format(suc.precision()))
        print("Total case : {}".format(suc.total()))

def fivefold(x_text, y):
    n_fold = 5
    size = len(x_text)
    fold_size = int(size / n_fold)
    for test_idx in range(n_fold):
        mid1 = test_idx*fold_size
        mid2 = (test_idx+1)*fold_size

        train_x = x_text[:mid1] + x_text[mid2:]
        train_y = np.concatenate([y[:mid1],y[mid2:]], axis=0)
        dev_x = x_text[mid1:mid2]
        dev_y = y[mid1:mid2]
        svm_phrase = Model()
        svm_phrase.train(train_x, train_y)
        print("Accuracy on train data: {}".format(svm_phrase.accuracy(train_x, train_y)))
        print("Accuracy on dev data: {}".format(svm_phrase.accuracy(dev_x, dev_y)))



if __name__ == "__main__":
    random.seed(0)

    pos_path = "..\\LRP\\data\\guardianC.txt"
    neg_path = "..\\LRP\\data\\guardianNC.txt"
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
            svm_phrase = Model()
            svm_phrase.train(x_text, y)
            #print("Accuracy on train data: {}".format(svm_phrase.accuracy(x_text, y)))

            answers = data_helpers.load_answer(test_data)
            linkl, test_text = zip(*test_data)
            print("Accuracy on contrv data: {}".format(svm_phrase.accuracy(test_text, np.ones([len(test_text)]))))
            svm_phrase.test_phrase(test_text, y, answers)

        else:
            fivefold(x_text, y)