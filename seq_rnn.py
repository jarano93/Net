#!usr/bin/python

import numpy as np
from pymod.rnn import RNN as RNN
from pymod.char import CharCodec as CCodec
import pymod.onehot as oh

iteration = 5

sample_size = 200

step_size = 5e-3
momentum = 5e-2

hid0 = 10
hid1 = 20

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()

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
