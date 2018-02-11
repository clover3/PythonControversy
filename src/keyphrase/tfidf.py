import pickle
import random
import collections
import math
import data_helpers
from nltk.tokenize import wordpunct_tokenize


def load_stopwords():
    s = set()
    f = open("..\\..\\data\\stopwords.dat", "r")
    for line in f:
        s.add(line.strip())
    return s

class keyphrase_tfidf:
    def __init__(self, split_no):
        ""
        self.split_no = split_no
        self.stopwords = load_stopwords()

    def tokenize(self, str):
        return wordpunct_tokenize(str)

    def train(self, data):
        tokened_corpus = [self.tokenize(s) for s in data]
        voca = collections.Counter()
        for tokens in tokened_corpus:
            for token in tokens:
                voca[token] += 1
        self.N = len(tokened_corpus)
        self.def_idf = math.log(self.N / 2)
        self.total_words = sum(voca.values())

        # convert document to bag of phrase
        phrase_doc_rep = []
        for doc in tokened_corpus:
            phrase_list = collections.Counter()
            for phrase_len in [1, 2, 3]:
                for i in range(0, len(doc)-phrase_len):
                    phrase = doc[i:i+phrase_len]
                    phrase_list[" ".join(phrase)] += 1
            phrase_doc_rep.append(phrase_list)

        self.phrase_df = collections.Counter()
        phrase_f = collections.Counter()
        for phrase_list in phrase_doc_rep:
            for phrase in phrase_list.keys():
                self.phrase_df[phrase] += 1
                phrase_f[phrase] += phrase_list[phrase]

        def meaningful(phrase):
            token_phrase = phrase.split(" ")
            if token_phrase[0] in self.stopwords or token_phrase[-1] in self.stopwords:
                return False
            for i, word in enumerate(token_phrase):
                try:
                    if i==0:
                        continue
                    prefix = " ".join(token_phrase[:i])
                    cur_phrase = " ".join(token_phrase[:i+1])
                    tf = voca[word]
                    p_b = tf / self.total_words
                    p_b_bar_a = phrase_f[cur_phrase] / phrase_f[prefix]
                    if p_b_bar_a < p_b * 2:
                        return False
                except ZeroDivisionError:
                    raise
            return True


        candidate = collections.Counter()
        for phrase in phrase_f.keys():
            if self.phrase_df[phrase]> 2 and meaningful(phrase):
                    candidate[phrase] += phrase_f[phrase]
        self.candidate_phrase = list([item[0] for item in candidate.most_common(3000)])
        pickle.dump(self.candidate_phrase, open("phrase3000_{}.pickle".format(split_no), "wb"))
        print("filtered phrase {}".format(len(self.candidate_phrase)))

    def get_idf(self, phrase):
        if phrase in self.phrase_df:
            df = self.phrase_df[phrase]
            return math.log(self.n_docs/df)
        else:
            return self.def_idf

    def analyze(self, data, answers):
        tokened_corpus = [self.tokenize(s) for s in data]
        self.n_docs = len(tokened_corpus)
        for doc_no, doc in enumerate(tokened_corpus):
            phrase_list = collections.Counter()
            for phrase_len in [1, 2, 3]:
                for i in range(0, len(doc)-phrase_len):
                    phrase = doc[i:i+phrase_len]
                    phrase_str = " ".join(phrase)
                    if phrase_str in self.candidate_phrase:
                        phrase_list[phrase_str] += 1

            priority_queue = collections.Counter()
            for phrase, tf in phrase_list.items():
                score = tf * self.get_idf(phrase)
                priority_queue[phrase] += score
            print("answer:\t{}".format(answers[doc_no]))
            for candi in priority_queue.most_common(10):
                print(candi)


if "__main__" == __name__ :
    random.seed(0)


    # Load data
    print("Loading data...")
    splits = pickle.load(open("splits.pickle", "rb"))
    for split_no, split in enumerate(splits):
        print("Split {}".format(split_no))
        x_text, y, test_data = split
        model = keyphrase_tfidf(split_no)
        model.train(x_text)
        answers = data_helpers.load_answer(test_data)
        linkl, test_text = zip(*test_data)
        #model.analyze(test_text, answers)

