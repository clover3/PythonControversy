
import pickle
import collections
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english', ignore_stopwords=True)



def print_key_value(l, k):
    for key, value in l[:k]:
        print("{} {}".format(key, value))

def left(l):
    r = []
    for k,v in l:
        r.append(k)
    return r

def makedict(l):
    d = dict()
    for k, v in l:
        if k not in d:
            d[k] = v
    return d

rev_stem_dict = dict()

def stem(word):
    result = stemmer.stem(word)
    rev_stem_dict[result] = word
    return result

def rev_stemp(stemmed):
    if stemmed in rev_stem_dict:
        return rev_stem_dict[stemmed]
    else:
        return stemmed

def rev_stem_phrase(phrase):
    return " ".join([rev_stemp(t) for t in phrase.split(" ")])

def merge_eval(split_no):
    middle_scores1 = pickle.load(open("middle.score{}.pickle".format(split_no), "rb"))
    middle_scores2 = pickle.load(open("middle.q.score{}.pickle".format(split_no), "rb"))
    answers = pickle.load(open("answer{}.pickle".format(split_no), "rb"))
    assert (len(answers) == len(middle_scores1))
    assert (len(answers) == len(middle_scores2))

    success_at = []
    cnn_success_at = []
    svm_success_at = []
    for middle_score1, middle_score2, answer in zip(middle_scores1, middle_scores2, answers):
        something = 0
        m_point = collections.Counter()

        for value, phrase in middle_score1:
            m_point[phrase] += value
        #print("-------- {} ------".format(answer))

        score1s = m_point.most_common()

        middle_score2.sort(key=lambda x:x[0], reverse=True)
        score2s = []
        rank = 1
        for value, key in middle_score2:
            phrase = " ".join([stem(token) for token in key])
            score2s.append((phrase, rank))
            rank = rank + 1
        score2s.sort(key=lambda x:x[1])

        sum_list = []
        rank1 = 1
        rank2dict = makedict(score2s)
        rank1dict = dict()

        for key, value in score1s:
            if key in rank2dict:
                rank2 = rank2dict[key]
            else:
                rank2 = 100
            if key not in rank1dict:
                rank1dict[key] = rank1
                sum_score = 1.0001/rank1 + 1/rank2
                sum_list.append((key, sum_score))
                rank1 = rank1 + 1

        for key, value in score2s:
            rank2 = rank2dict[key]
            if key not in rank1dict:
                rank1 = 100
                sum_score = 1.0001 / rank1 + 1 / rank2
                sum_list.append((key, sum_score))

        answer_key = " ".join([stem(token) for token in answer.split(" ")])
        #print(answer_key)
        rank_at_1 = rank1dict[answer_key] if answer_key in rank1dict else 100
        rank_at_2 = rank2dict[answer_key] if answer_key in rank2dict else 100
        if rank_at_1 > 100 :
            rank_at_1 = 100
        if rank_at_2 > 100:
            rank_at_2 = 100

        cnn_success_at.append(rank_at_1)
        svm_success_at.append(rank_at_2)
        print("{}\t{}\t{}".format(answer, rank_at_1, rank_at_2))
        sum_list.sort(key=lambda x: x[1], reverse=True)
        #print("--- final -----")
        suc_idx = 51
        for idx, (key, value) in enumerate(sum_list[:51]):
            if key == answer_key:
                suc_idx = idx
                break
        success_at.append(suc_idx)


        cnt =0
        for k1,k2,k3 in zip(left(score1s), left(score2s), left(sum_list)):
            if cnt >= 10:
                break
            k1 = rev_stem_phrase(k1)
            k2 = rev_stem_phrase(k2)
            k3 = rev_stem_phrase(k3)
            #print("{}\t{}\t{}".format(k1,k2,k3))
            cnt = cnt + 1

    total =len(success_at)
    #print("{} cases".format(total))
    assert(len(cnn_success_at) == len(answers))
    assert (len(svm_success_at) == len(answers))
    print("split {}".format(split_no))
    print("Ensemble")
    dict_ensemble = dict()
    for k in range(1,21):
        suc = sum(1 if i <k else 0 for i in success_at)
        dict_ensemble[k] = suc
        print("Suc@{}\t{}".format(k, suc/ total))

    print("CNN")
    dict_cnn = dict()
    for k in range(1,21):
        cnn_suc = sum(1 if i<=k else 0 for i in cnn_success_at)
        dict_cnn[k] = cnn_suc
        print("Suc@{}\t{}".format(k, cnn_suc/ total))

    print("SVM")
    dict_svm = dict()
    for k in range(1,21):
        suc = sum(1 if i<=k else 0 for i in svm_success_at)
        dict_svm[k] = suc
        print("Suc@{}\t{}".format(k, suc/ total))

    dict_systems = {
        "ensemble":dict_ensemble,
        "SVM": dict_svm,
        "CNN": dict_cnn,
    }
    return dict_systems


runs = []
for split_no in range(3):
    runs.append(merge_eval(split_no))

system_list = ["SVM", "CNN", "ensemble"]
for system in system_list:
    print()
for k in range(1,21):
    print("{}\t".format(k), end="")

    scores = dict()
    for system in system_list:
        score = 0
        for split_no in range(3):
            score = score + runs[split_no][system][k]
        scores[system] = score /124

    print("{}\t{}\t{}".format(scores["SVM"], scores["CNN"], scores["ensemble"]))

