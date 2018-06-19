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

def get_rr_list(target, f_textrank= True):
    if target == "CNN":
        pickle_form = "middle_bestrun\\middle.score{}.pickle"
        mode = "CNN"
    elif target == "CNN_wo_phrase":
        pickle_form = "middle_bestrun\\without_phrase\\middle.score{}.pickle"
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
        if f_textrank:
            pickle_form = "middle_bestrun\\with_textrank\\middle.lm.score{}.pickle"
        else:
            pickle_form = "middle_bestrun\\no_textrank\\middle.lm.score{}.pickle"
        mode = "LinSVM"
    elif target == "LM_wo_phrase":
        pickle_form = "middle_bestrun\\without_phrase\\middle.lm.score{}.pickle"
        mode = "LinSVM"
    elif target == "Phrase":
        mode = "Phrase"
        pickle_form = "middle_bestrun\\middle.q.score{}.pickle"
    elif target == "Nothing":
        mode = "Nothing"
        pickle_form = "middle_bestrun\\middle.q.score{}.pickle"

    def stem_textrank(raw_textrank):
        textrank = collections.Counter()
        for phrase, value in raw_textrank.items():
            key = " ".join([stem(token) for token in phrase.split(" ")])
            textrank[key] += value
        return textrank

    def merge_eval(pickle_form, split_no):
        path = pickle_form.format(split_no)
        middle_scores = pickle.load(open(path, "rb"))
        tr_path = "..\\keyphrase\\textrank_{}.pickle".format(split_no)
        textrank_list = pickle.load(open(tr_path, "rb"))

        answers = pickle.load(open("answer{}.pickle".format(split_no), "rb"))
        assert (len(answers) == len(middle_scores))
        success_at = []

        idx = 0
        for middle_score, answer in zip(middle_scores, answers):
            textrank = stem_textrank(textrank_list[idx])
            idx += 1
            answer_key = " ".join([stem(token) for token in answer.split(" ")])
            rank1dict = dict()
            if mode == "CNN":
                m_point = collections.Counter()
                for value, phrase in middle_score:
                    m_point[phrase] += value
                # print("-------- {} ------".format(answer))

                for phrase, score in m_point.items():
                    factor = textrank[phrase] if f_textrank else 1
                    m_point[phrase] = score * factor

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

                if textrank is not None:
                    new_score = []
                    for value, phrase in middle_score:
                        system_answer = stem_phrase(" ".join(phrase))
                        factor = textrank[system_answer] if f_textrank else 1
                        new_score.append((value * factor, system_answer))
                    new_score.sort(key=lambda x:x[0], reverse=True)
                    for value, phrase in new_score:
                        if phrase not in rank1dict:
                            rank1dict[phrase] = rank1
                        rank1 = rank1 + 1

                else:
                    for value, phrase in middle_score:
                        system_answer = stem_phrase(" ".join(phrase))
                        if system_answer not in rank1dict:
                            rank1dict[system_answer] = rank1
                        rank1 = rank1 + 1
            elif mode == "Nothing":
                scores = sorted(list(textrank.items()), key=lambda x:x[1], reverse=True)
                rank1 = 1
                for key, value in scores:
                    if key not in rank1dict:
                        rank1dict[key] = rank1
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

def get_acc(target, f_textrank= True):
    rr = get_rr_list(target, f_textrank)
    acc = sum([1 if r == 1 else 0 for r in rr]) / len(rr)
    return acc

def get_acc_list(target, f_textrank= True):
    rr = get_rr_list(target, f_textrank)
    return list([1 if r == 1 else 0 for r in rr])

def get_mrr(target, f_textrank= True):
    rr = get_rr_list(target, f_textrank)
    mrr = sum(rr) / len(rr)
    return mrr

def stat_test():

    from scipy import stats
    
    def pair_test(r1, r2):
        _, p = stats.ttest_rel(r1, r2)
        if p < 0.05:
            return True
        else:
            return p

    def analyze_significance(get_measure_list):
        print("wo phrase")
        r_lm_wo = get_measure_list("LM_wo_phrase", False)
        r_cnn_wo = get_measure_list("CNN_wo_phrase", False)
        print("LM-CNN")
        print(pair_test(r_cnn_wo, r_lm_wo))

        print("w phrase")
        r_first = get_measure_list("First")
        r_random = get_measure_list("Random")
        r_cnn = get_measure_list("CNN", False)
        r_phrase = get_measure_list("Phrase", False)
        r_lm = get_measure_list("LM", False)
        print("First - Random {}".format(pair_test(r_first, r_random)))

        print("LM - (wo) {}".format(pair_test(r_lm, r_lm_wo)))
        print("LM - Random {}".format(pair_test(r_lm, r_random)))
        print("LM - First{}".format(pair_test(r_lm, r_first)))
        print("LM - CNN{}".format(pair_test(r_lm, r_cnn)))

        print("CNN - (wo) {}".format(pair_test(r_cnn, r_cnn_wo)))
        print("CNN - Random {}".format(pair_test(r_cnn, r_random)))
        print("CNN - First {}".format(pair_test(r_cnn, r_first)))


        print("Phrase - First {}".format(pair_test(r_phrase, r_first)))
        print("Phrase - Random {}".format(pair_test(r_phrase, r_random)))
        print("Phrase - LM {}".format(pair_test(r_phrase, r_lm)))
        print("Phrase - CNN {}".format(pair_test(r_phrase, r_cnn)))

        r_lm_tr = get_measure_list("LM", True)
        r_cnn_tr = get_measure_list("CNN", True)
        r_phrase_tr = get_measure_list("Phrase", True)
        r_tr = get_measure_list("Nothing", True)
        print("-- Text Rank ----")
        print("LM - (wo) {}".format(pair_test(r_lm_tr, r_lm)))
        print("LM - TR {}".format(pair_test(r_lm_tr, r_tr)))

        print("Phrase - (wo) {}".format(pair_test(r_phrase_tr, r_phrase)))
        print("Phrase - TR {}".format(pair_test(r_phrase_tr, r_tr)))
        print("Phrase - LM {}".format(pair_test(r_phrase_tr, r_lm_tr)))

        print("CNN - (wo) {}".format(pair_test(r_cnn_tr, r_cnn)))
        print("CNN - TR {}".format(pair_test(r_cnn_tr, r_tr)))
        print("CNN - LM {}".format(pair_test(r_cnn_tr, r_lm_tr)))
        print("CNN - Phrase {}".format(pair_test(r_cnn_tr, r_phrase_tr)))

    print("---MRR----")
    analyze_significance(get_rr_list)
    print("---ACC----")
    analyze_significance(get_acc_list)


def print_all_mrr():
    mrr_dict = {}
    for target in ["Random", "First", "CNN", "Phrase", "LM", "Nothing"]:
        mrr = get_mrr(target, True)
        mrr_dict[target] = mrr
        acc = get_acc(target, True)
        print(target)
        print("{0:.2f}".format(acc))
        print("{0:.2f}".format(mrr))



if __name__ == "__main__":
    stat_test()