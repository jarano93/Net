#!usr/bin/python

import numpy as np
from pymod.rec import RNN as RNN
from pymod.char import CharCodec as CCodec

f = open('twcset.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

# modelparams
h0_len = 55
h1_len = 70

print "%d unique characters in dataset\n\n" % uni_chars

rnn = RNN(uni_chars, h0_len, h1_len)
# rnn.set_freq(50)
rnn.set_sample_len(800)
rnn.set_rollback(200)
# rnn.set_clip(10)
rnn.set_codec(cc)
rnn.train_TOL(int_dataset, 5)
rnn.train_N(int_dataset, 10000)
