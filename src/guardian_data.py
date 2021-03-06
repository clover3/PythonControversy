import os
import json
import pickle
import math
from clover_lib import *

def load_data(corpus_path, label_path, format, idx2wordPath=None, query2idx=None):

    data_cache = os.path.join("cache",format+".pickle")
    #if os.path.isfile(data_cache):
    #    print("Loaded data from pickle")
    #    return pickle.load(open(data_cache,"rb"))
    print("Loading data")

    f = open(corpus_path, "r")

    def parse_line(line):
        if format == "unigram" or format == "bigram":
            tokens = line.split("\t")
            article_id = tokens[0]
            query = tokens[1]
            relevance = math.exp(float(tokens[2]))
            return article_id, query, relevance
        else:
            raise Exception("format is not expected one")

    voca = set()
    articles = dict()
    if not idx2wordPath:
        lines = f.readlines()
        for line in lines :
            article_id, query, relevance = parse_line(line)
            voca.add(query)

            if article_id not in articles:
                articles[article_id] = list()

            articles[article_id].append((query, relevance))
    else:
        idx2word = dict()
        for line in open(idx2wordPath, "r"):
            tokens = line.strip().split(" ")
            idx2word[int(tokens[1])] = tokens[0]
        print("{} word voca".format(len(idx2word)))
        print("Reading data...")
        line = f.readline()
        while line:
            tokens= line.split("\t")
            article_id = tokens[0]
            articles[article_id] = list()
            count = int(tokens[1])
            for i in range(count):
                tokens = f.readline().split(" ")
                query = idx2word[int(tokens[0])]
                voca.add(query)
                score = float(tokens[1])
                articles[article_id].append((query, score))

            line = f.readline()


    labels = dict()
    for line in open(label_path).readlines():
        tokens = line.split(" ")
        article_id = tokens[0]
        score = int(tokens[1])
        labels[article_id] = score

    if query2idx is None:
        print("Building query2idx")
        idx = 1
        query2idx = dict()
        for query in voca:
            query2idx[query] = idx
            idx = idx + 1

    data = []
    missingCount = 0
    wordMiss = FailCounter()
    for article_id in articles.keys():
        word_rel_list = articles[article_id]
        query_list = []
        rel_list = []
        for (query,rel) in word_rel_list:
            if query in query2idx:
                query_list.append(query2idx[query])
                rel_list.append(rel)
                wordMiss.suc()
            else:
                wordMiss.fail()

        if article_id in labels:
            label = labels[article_id]

            entry = (query_list, rel_list, label, article_id)
            data.append(entry)
        else:
            missingCount = missingCount +1
    print("Missing {} article label".format(missingCount))
    print("{} of the words are missing".format(1-wordMiss.precision()))
    voca_size = len(voca)+1
    r = data, voca_size, query2idx

    pickle.dump(r, open(data_cache, "wb"))
    return r


def unigram2bigram(query2idx):
    unigram_path = os.path.join("cache","unigram.pickle")
    data, voca_size, word2idx = pickle.load(open(unigram_path, "rb"))

    emb_path = os.path.join("result", "we_arr.pickle")
    we_arr = pickle.load(open(emb_path, "rb"))
    emb_dict_arr = []
    for we in we_arr:
        emb_dict = dict()
        for query in query2idx.keys():
            tokens = query.split(" ")
            match = 0
            weight = 0
            for token in tokens:
                if token in word2idx:
                    weight = weight + we[word2idx[token]]
                    match = match + 1
            if match > 1:
                avg_weight = weight / match
                emb_dict[query2idx[query]] = avg_weight
        emb_dict_arr.append(emb_dict)
    return emb_dict_arr


def load_article(dir_path):
    articles = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path) :
            print(path)
            f = open(path, encoding='utf-8')
            j = json.load(f)
            for j_article in j['response']['results']:
                id = j_article['id']
                body_text = j_article['fields']['bodyText']
                short_url = j_article['fields']['shortUrl']
                articles.append((id, short_url, body_text))
    return articles
