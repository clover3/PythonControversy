import os
import pickle

def load_data(corpus_path, label_path):

    data_cache = os.path.join("cache","data1.pickle")
    #if os.path.isfile(data_cache):
    #    print("Loaded data from pickle")
    #    return pickle.load(open(data_cache,"rb"))
    print("Loading data")

    f = open(corpus_path, "r")
    lines = f.readlines()

    voca = set()
    articles = dict()
    for line in lines :
        tokens = line.split("\t")
        article_id = tokens[0]
        word = tokens[1]
        relevance = float(tokens[2])
        voca.add(word)

        if article_id not in articles:
            articles[article_id] = list()

        articles[article_id].append((word, relevance))

    labels = dict()
    for line in open(label_path).readlines():
        tokens = line.split(" ")
        article_id = tokens[0]
        score = int(tokens[1])
        labels[article_id] = score

    idx = 1
    word2idx = dict()
    for word in voca:
        word2idx[word] = idx
        idx = idx +1

    data = []
    missingCount = 0
    for article_id in articles.keys():
        word_rel_list = articles[article_id]
        word_list = []
        rel_list = []
        for (word,rel) in word_rel_list:
            word_list.append(word2idx[word])
            rel_list.append(rel)

        if article_id in labels:
            label = labels[article_id]

            entry = (word_list, rel_list, label)
            data.append(entry)
        else:
            missingCount = missingCount +1
    print("Missing {} article label".format(missingCount))
    voca_size = len(voca)+1
    r = data, voca_size, word2idx
    pickle.dump(r, open(data_cache, "wb"))
    return r


