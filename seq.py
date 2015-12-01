#!usr/bin/python

import numpy as np
from pymod.lstm import LSTM as LSTM
from pymod.char import CharCodec as CCodec
import pymod.onehot as oh

sample_size = 200

f = open('twcshort.txt', 'r')
str_dataset = f.read()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence(str_dataset)

num_uniques = cc.length()
dataset = np.zeros((seq_length, num_uniques))
for i in xrange(seq_length):
    dataset[i,:] = oh.hot(int_dataset[i], num_uniques)

lstm = LSTM(num_uniques)

lstm.seq_loss(dataset)
while True:
    lstm.train_N(dataset, 100)
    out = lstm.sample(sample_size)
    res = []
    for i in xrange(sample_size):
        res.append(oh.key(out[i]))
    print cc.string(res)
