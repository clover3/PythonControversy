import pickle
import collections
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english', ignore_stopwords=True)
rev_stem_dict = dict()

def stem(word):
    result = stemmer.stem(word)
    rev_stem_dict[result] = word
    return result


## CNN : Stemed phrase, value
## LinSVM : not stemmed ranked phrase
mode = "CNN"

def stem_phrase(phrase):
    return " ".join([stem(token) for token in phrase.split(" ")])

def merge_eval(pickle_form, split_no):
    middle_scores = pickle.load(open(pickle_form.format(split_no), "rb"))
    answers = pickle.load(open("answer{}.pickle".format(split_no), "rb"))
    print(len(middle_scores))
    assert (len(answers) == len(middle_scores))
    success_at = []

    for middle_score, answer in zip(middle_scores, answers):
        answer_key = " ".join([stem(token) for token in answer.split(" ")])
        rank1dict = dict()
        if mode == "CNN":
            m_point = collections.Counter()
            for value, phrase in middle_score:
                m_point[phrase] += value
            # print("-------- {} ------".format(answer))

            score1s = m_point.most_common()
            rank1 = 1
            for key, value in score1s:
                if key not in rank1dict:
                    rank1dict[key] = rank1
                rank1 = rank1 + 1
        elif mode == "LinSVM":
            rank1 = 1
            print("answer : " + answer_key)
            for phrase in middle_score:
                system_answer = stem_phrase(phrase)
                if rank1 == 1:
                    print("sys : " + phrase)
                if system_answer not in rank1dict:
                    rank1dict[system_answer] = rank1
                rank1 = rank1 + 1

        rank = rank1dict[answer_key] if answer_key in rank1dict else 100
        success_at.append(rank-1)
    score = dict()
    total =len(success_at)
    for k in range(1,21):
        suc = sum(1 if i <k else 0 for i in success_at)
        score[k] = suc
        print("Suc@{}\t{}".format(k, suc/ total))
    return score

pickle_form = "middle_cnn_1\\middle.score{}.pickle"

runs = []
for split_no in range(3):
    runs.append(merge_eval(pickle_form, split_no))

for k in range(1,21):
    #print("{}\t".format(k), end="")
    score = 0
    for split_no in range(3):
        score = score + runs[split_no][k]
    print(score / 124)
