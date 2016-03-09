#!usr/bin/python

import numpy as np
from pymod.rec import RNN as RNN
from pymod.char import CharCodec as CCodec

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence(str_dataset)
uni_chars = cc.length()

# modelparams
h0_len = 50
h1_len = 100

rnn = RNN(uni_chars, h0_len, h1_len)
rnn.set_codec(cc)
rnn.train_N(int_dataset, 1000000)
