import numpy as np
import os
from time import strftime, localtime

root_dir = "debugs"


class Logger:
    def __init__(self):
        self.unit_dict = dict()
        folder_name = strftime("%d %b %Y %H %M %S", localtime())
        self.path = root_dir + "\\" + folder_name
        os.makedirs(self.path)

        np.set_printoptions(precision=5, suppress=True)

    def set_prefix(self, prefix):
        self.prefix = prefix
        for unit in self.unit_dict.values():
            unit.set_prefix(prefix)


    def print(self, name, obj):
        if name not in self.unit_dict:
            self.unit_dict[name] = LoggingUnit(self.path, name)

        self.unit_dict[name].print(self.prefix)
        self.unit_dict[name].print(obj)


class LoggingUnit:
    def __init__(self, path, name):
        self.f = open(path + "\\" + name, "w+")
        self.id = 0
        self.f.write("File Beggining --------------------\n")
        self.prefix = None

    def set_prefix(self, prefix):
        self.prefix = prefix

    def print(self, obj):
        self.log(obj.__str__())

    def log(self, msg):
        self.f.write("------------Log {}----------------\n".format(self.id))
        if self.prefix:
            self.f.write(self.prefix)
        self.f.write(msg + "\n")
        self.f.write("----------------------------------\n")
        self.id += 1


def print_word_embedding(word2idx, we_arr):
    we = we_arr[0]

    print(type(we))

    for word in word2idx.keys():
        idx = word2idx[word]
        print("{} {} {}".format(idx, word, we[idx]))

