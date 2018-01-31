from nltk.tokenize import wordpunct_tokenize
import collections
import pickle

def load_stopwords():
    s = set()
    f = open("..\\..\\data\\stopword2.txt", "r")
    for line in f:
        s.add(line.strip())
    print("Loaded {} stopwords".format(len(s)))
    return s

class PhraseGen:
    def __init__(self):
        ""

    def tokenize(self, str):
        return wordpunct_tokenize(str)

    def phrase_gen(self, x):
        print("Tokenizing")
        self.tokened_corpus = [self.tokenize(s) for s in x]
        self.n_docs = len(self.tokened_corpus)

        self.df = collections.Counter()
        voca = collections.Counter()
        self.doc_sparse = []
        for tokens in self.tokened_corpus:
            doc_sparse = collections.Counter()
            for token in tokens:
                voca[token] += 1
                doc_sparse[token] += 1
            for key in doc_sparse:
                self.df[key] += 1
            self.doc_sparse.append(doc_sparse)
        self.total_words = sum(voca.values())
        print("")
        return self.generate_feature_phrases(self.tokened_corpus, voca, self.total_words)

    def generate_feature_phrases(self, tokened_corpus, voca, total_words):
        n_docs = len(tokened_corpus)
        stopwords = load_stopwords()
        def all_stopword(phrase):
            return all([token in stopwords for token in phrase])

        phrase_doc_rep = []
        for doc in tokened_corpus:
            phrase_list = collections.Counter()
            for phrase_len in [1, 2, 3]:
                for i in range(0, len(doc)-phrase_len):
                    phrase = doc[i:i+phrase_len]
                    phrase_list[" ".join(phrase)] += 1
            phrase_doc_rep.append(phrase_list)

        phrase_df = collections.Counter()
        phrase_f = collections.Counter()
        for phrase_list in phrase_doc_rep:
            for phrase in phrase_list.keys():
                phrase_df[phrase] += 1
                phrase_f[phrase] += phrase_list[phrase]

        print("All phrase {}".format(len(phrase_df)))

        def meaningful(phrase):
            token_phrase = phrase.split(" ")
            if token_phrase[0] in stopwords or token_phrase[-1] in stopwords:
                return False
            if all_stopword(token_phrase):
                return False
            for i, word in enumerate(token_phrase):
                try:
                    if i==0:
                        continue
                    prefix = " ".join(token_phrase[:i])
                    cur_phrase = " ".join(token_phrase[:i+1])
                    tf = voca[word]
                    p_b = tf / total_words
                    p_b_bar_a = phrase_f[cur_phrase] / phrase_f[prefix]
                    if p_b_bar_a < p_b * 2:
                        return False
                except ZeroDivisionError:
                    raise
            return True


        def very_meaningful(phrase):
            token_phrase = phrase.split(" ")
            if all_stopword(phrase):
                return False
            for i, word in enumerate(token_phrase):
                try:
                    if i==0:
                        continue
                    prefix = " ".join(token_phrase[:i])
                    cur_phrase = " ".join(token_phrase[:i+1])
                    tf = voca[word]
                    p_b = tf / total_words
                    p_b_bar_a = phrase_f[cur_phrase] / phrase_f[prefix]
                    if p_b_bar_a < p_b * 10:
                        return False
                except ZeroDivisionError:
                    raise

            return True

        candidate_phrase = []
        for phrase in phrase_f.keys():
            if phrase_df[phrase]> 3 and meaningful(phrase):
                candidate_phrase.append(phrase)
        print("filtered phrase {}".format(len(candidate_phrase)))
        return set(candidate_phrase)


splits = pickle.load(open("splits.pickle", "rb"))
for split_no, split in enumerate(splits):
    x_text, y, test_data = split
    generator = PhraseGen()
    phrases = list(generator.phrase_gen(x_text))
    pickle.dump(phrases, open("phrase{}.pickle".format(split_no), "wb"))
