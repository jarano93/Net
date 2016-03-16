#!/usr/bin/python

import numpy as np

class CharCodec():
    def __init__(self, string):
        self.string = string
        self.chars = list(set(string))
        self.num_chars = len(self.chars)
        self.charkeys = dict(zip(self.chars, range(self.num_chars)))
        self.intkeys = dict(zip(range(self.num_chars), self.chars))

    # converts a number to a character
    def char(self, num):
        try:
            return self.intkeys[num]
        except KeyError:
            return 'err'

    # converts a character to a number
    def num(self, char):
        return self.charkeys[char]

    def length(self):
        return self.num_chars

    def stringify(self, num_array):
        res_str = [''] * len(num_array)
        for i in xrange(len(num_array)):
            res_str[i] = self.char(num_array[i])
        return ''.join(res_str)
        
    def sequence(self):
        str_len = len(self.string)
        seq = np.zeros(str_len)
        for i in xrange(str_len):
            seq[i] = self.num(self.string[i])
        return seq
