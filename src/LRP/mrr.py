import collections
import pickle

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english', ignore_stopwords=True)
rev_stem_dict = dict()

def stem(word):
    result = stemmer.stem(word)
    rev_stem_dict[result] = word
    return result

def stem_phrase(phrase):
    return " ".join([stem(token) for token in phrase.split(" ")])


## CNN : Stemed phrase, value
## LinSVM : not stemmed ranked phrase

target = "CNN"

def get_rr_list(target):

    if target == "CNN":
        pickle_form = "middle_bestrun\\middle.score{}.pickle"
        mode = "CNN"
    elif target == "Random":
        pickle_form = "middle_bestrun\\middle.r.score{}.pickle"
        mode = "LinSVM"
    elif target == "First":
        pickle_form = "middle_bestrun\\middle.f.score{}.pickle"
        mode = "LinSVM"
    elif target == "LinSVM":
        pickle_form = "middle_bestrun\\middle.u.score{}.pickle"
        mode = "LinSVM"
    elif target == "LM":
        pickle_form = "middle_bestrun\\middle.lm.score{}.pickle"
        mode = "LinSVM"
    elif target == "Phrase":
        mode = "Phrase"
        pickle_form = "middle_bestrun\\middle.q.score{}.pickle"

    def merge_eval(pickle_form, split_no):
        path = pickle_form.format(split_no)
        middle_scores = pickle.load(open(path, "rb"))
        answers = pickle.load(open("answer{}.pickle".format(split_no), "rb"))
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
                for phrase in middle_score:
                    system_answer = stem_phrase(phrase)
                    if system_answer not in rank1dict:
                        rank1dict[system_answer] = rank1
                    rank1 = rank1 + 1
            elif mode == "Phrase":
                rank1 = 1
                for value, phrase in middle_score:
                    system_answer = stem_phrase(" ".join(phrase))
                    if system_answer not in rank1dict:
                        rank1dict[system_answer] = rank1
                    rank1 = rank1 + 1

            rank = rank1dict[answer_key] if answer_key in rank1dict else 100
            success_at.append(rank)
        return success_at

    runs = []
    for split_no in range(3):
        runs += merge_eval(pickle_form, split_no)
    suc5_raw = [1 if rank<11 else 0 for rank in runs]
    suc5 = sum(suc5_raw) / len(suc5_raw)
    #print("suc5 : {}".format(suc5))
    rr = [1/rank for rank in runs]
    return rr

def get_acc(target):
    rr = get_rr_list(target)
    acc = sum([1 if r == 1 else 0 for r in rr]) / len(rr)
    return acc

def get_acc_list(target):
    rr = get_rr_list(target)
    return list([1 if r == 1 else 0 for r in rr])

def get_mrr(target):
    rr = get_rr_list(target)
    mrr = sum(rr) / len(rr)
    return mrr

def stat_test():
    from scipy import stats
    r_cnn = get_rr_list("CNN")
    r_phrase = get_rr_list("Phrase")
    r_first = get_rr_list("First")
    r_random = get_rr_list("Random")
    r_lm = get_rr_list("LM")
    print("Phrase-CNN")
    print(stats.ttest_rel(r_phrase, r_cnn))
    print("Phrase-LM")
    print(stats.ttest_rel(r_phrase, r_lm))

    print("CNN-first")
    print(stats.ttest_rel(r_cnn, r_first))

    print("CNN-LM")
    print(stats.ttest_rel(r_cnn, r_lm))

    print("LM-first")
    print(stats.ttest_rel(r_lm, r_first))
    print("First-random")
    print(stats.ttest_rel(r_first, r_random))

    print("---ACC----")
    acc_cnn = get_acc_list("CNN")
    acc_phrase = get_acc_list("Phrase")
    acc_first = get_acc_list("First")
    acc_random = get_acc_list("Random")
    acc_lm = get_acc_list("LM")
    print("Phrase-CNN")
    print(stats.ttest_rel(acc_phrase, acc_cnn))
    print("Phrase-LM")
    print(stats.ttest_rel(acc_phrase, acc_lm))

    print("CNN-first")
    print(stats.ttest_rel(acc_cnn, acc_first))

    print("CNN-LM")
    print(stats.ttest_rel(acc_cnn, acc_lm))

    print("LM-first")
    print(stats.ttest_rel(acc_lm, acc_first))

    print("First-random")
    print(stats.ttest_rel(acc_first, acc_random))


def print_all_mrr():
    print("MRR")
    mrr_dict = {}
    for target in ["Random", "First", "CNN", "Phrase", "LM"]:
        mrr = get_mrr(target)
        mrr_dict[target] = mrr
        acc = get_acc(target)
        print("MRR\t{}\t{}".format(target, mrr))
        print("ACC\t{}\t{}".format(target, acc))


if __name__ == "__main__":
    stat_test()