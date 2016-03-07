#!usr/bin/python

import numpy as np
from pymod.rnn import RNN as RNN
from pymod.char import CharCodec as CCodec
import pymod.onehot as oh

iteration = 100

sample_size = 200

step_size = 1e-2
momentum = 5e-1

hid0 = 200
hid1 = 300

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence(str_dataset)

num_uniques = cc.length()
dataset = np.zeros((seq_length, num_uniques))
for i in xrange(seq_length):
    dataset[i,:] = oh.hot(int_dataset[i], num_uniques)

rnn = RNN(num_uniques, hid0, hid1)

rnn.seq_loss(dataset)

while True:
    rnn.train_N(dataset, step_size, momentum, iteration)
    out = rnn.sample(sample_size)
    res = []
    for i in xrange(sample_size):
        res.append(oh.key(out[i]))
    print cc.string(res)
