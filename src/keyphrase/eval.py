import collections
from nltk.tokenize import wordpunct_tokenize
import pickle
import random
from keyphrase.textrank import *
import data_helpers

def tokenize(str):
    return wordpunct_tokenize(str)

def analyze(data, answers, model):
    candidate_phrase = pickle.load(open("phrase3000_{}.pickle".format(split_no), "rb"))

    tokened_corpus = [tokenize(s) for s in data]
    n_docs = len(tokened_corpus)
    rank = []
    for doc_no, doc in enumerate(tokened_corpus):
        phrase_list = collections.Counter()
        for phrase_len in [1, 2, 3]:
            for i in range(0, len(doc) - phrase_len):
                phrase = doc[i:i + phrase_len]
                phrase_str = " ".join(phrase)
                if phrase_str in candidate_phrase:
                    phrase_list[phrase_str] += 1

        p_score = model.run(doc)
        def score_fn(phrase):
            return sum([p_score[word] for word in phrase.split(" ")])

        priority_queue = collections.Counter()
        for phrase, tf in phrase_list.items():
            priority_queue[phrase] += score_fn(phrase)
        rank.append(priority_queue)
        print("answer:\t{}".format(answers[doc_no]))
        for candi in priority_queue.most_common(10):
            print(candi)
    pickle.dump(rank, open("textrank_{}.pickle".format(split_no), "wb"))

if "__main__" == __name__:
    random.seed(0)

    # Load data
    print("Loading data...")
    splits = pickle.load(open("splits.pickle", "rb"))
    for split_no, split in enumerate(splits):
        print("Split {}".format(split_no))

        x_text, y, test_data = split
        answers = data_helpers.load_answer(test_data)
        linkl, test_text = zip(*test_data)
        model = TextRank(test_text)
        analyze(test_text, answers, model)

