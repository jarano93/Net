#!usr/bin/python

import numpy as np

class CharCodec():
    def __init__(self, string):
        self.chars = list(set(string))
        self.num_chars = len(self.chars)
        self.charkeys = dict(zip(self.chars, range(self.num_chars)))
        self.intkeys = dict(zip(range(self.num_chars), self.chars))

    def char(self, num):
        return self.intkeys[num]

    def num(self, char):
        return self.charkeys[char]

    def length(self):
        return self.num_chars

    def string(self, num_array):
        res_str = [''] * len(num_array)
        for i in xrange(len(num_array)):
            res_str[i] = self.char(num_array[i])
        return ''.join(res_str)
        
    def sequence(self, string):
        seq = np.zeros(len(string))
        for i in xrange(len(string)):
            seq[i] = self.num(string[i])
        return seq
