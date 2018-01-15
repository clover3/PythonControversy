import pickle
import os.path


def load_side_articles():
    path = "C:\work\Code\Isidewith\\article_plain.pickle"
    articles = pickle.load(open(path, "rb"))
    # dict of key -> content
    # content contains newline(\n)

    data = []
    for (link, text) in articles.items():
        data.append((link,text))
    return data


def extract_text(articleList):
    # list[(link,text)
    data = []
    for (link, text) in articleList:
        data.append(text)
    return data