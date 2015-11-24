#!usr/bin/python

import numpy as np

class CharCodec():
    def __init__(self, string):
        self.chars = list(set(string)))
        self.num_chars = len(self.chars)
        self.charkeys = dict(zip(self.chars, self.num_chars))
        self.intkeys = dict(zip(self.num_chars, self.chars))

    def char(self, num):
        return self.intkeys[val]

    def num(self, char):
        return self.charkeys[char]

    def length(self):
        return self.num_chars
