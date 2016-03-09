#!usr/bin/python

from pymod.rec import RNN as RNN
from pymod.char import CharCodec as CCodec

f = open('twc_clean.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

# modelparams
h0_len = 120
h1_len = 160
w_scale = 1e-1

print "%d unique characters in dataset\n\n" % uni_chars

rnn = RNN(uni_chars, h0_len, h1_len, w_scale, True, True)
# rnn.set_freq(50)
rnn.set_sample_len(800)
rnn.set_rollback(200)
# rnn.set_clip(10)
rnn.set_codec(cc)
rnn.train_N(int_dataset, 1000000)
