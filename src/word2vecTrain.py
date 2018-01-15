from gensim.models.word2vec import Word2Vec
import pickle
import re

def tokenize(sentence):
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    tokens = clean_str(sentence).split(" ")
    return tokens

def load_guardian_sentences():
    all_sentences = []
    articles = pickle.load(open("2016articles.pickle","rb"))
    for (id, short_url, body_text) in articles:
        sentences = body_text.split("\n")
        all_sentences.extend([tokenize(s) for s in sentences])
    return all_sentences


sentences = ["There are also alternative routes to install",
    "If you have downloaded and unzipped the tar.gz source for gensim (or you’re installing gensim from github), you can run",
    "to install gensim into your site-packages folder.",
    "If you wish to make local changes to the gensim code (gensim is, after all, a package which targets research prototyping and modifications), a preferred way may be installing with"
    ]

sentences = list([tokenize(s) for s in sentences])
model = Word2Vec(load_guardian_sentences(), sg=1, size=100, window=10, min_count=10, workers=4)
model.save('mymodel')
