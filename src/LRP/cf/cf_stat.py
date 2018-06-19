from scipy import stats
import pickle
import numpy as np

def load(path):
    data = pickle.load(open(path, "rb"))
    return np.concatenate(data).astype(int)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def load_all(format):
    l= []
    for i in range(3):
        l.append(load(format.format(i)))
    return np.concatenate(l)

def gen_rand(l):
    return np.random.randint(2, size=l)

def gen_rand_label(l, label):
    return np.random.randint(2, size=l)



def stat_test():
    cf_cnn = load_all("cf_cnn_{}.pickle")
    cf_lm = load_all("cf_lm_{}.pickle")
    cf_phrase = load_all("cf_phrase_{}.pickle")

    cf_cnn_conf = load_all("cf_cnn_cont_{}.pickle")
    cf_lm_conf = load_all("cf_lm_cont_{}.pickle")
    cf_phrase_conf = load_all("cf_phrase_cont_{}.pickle")
    cf_random_conf = gen_rand(len(cf_phrase_conf))

    print("cnn : {}".format(mean_confidence_interval(cf_cnn)))
    print("lm: {}".format(mean_confidence_interval(cf_lm)))
    print("phrase: {}".format(mean_confidence_interval(cf_phrase)))

    print("cnn_conf: {}".format(mean_confidence_interval(cf_cnn_conf)))
    print("lm_conf: {}".format(mean_confidence_interval(cf_lm_conf)))
    print("phrase_conf: {}".format(mean_confidence_interval(cf_phrase_conf)))
    print("random_conf: {}".format(mean_confidence_interval(cf_random_conf)))



stat_test()