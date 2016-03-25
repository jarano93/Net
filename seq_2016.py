#!/usr/bin/python 

from pymod.rnn import RNN as RNN
from pymod.char import CharCodec as CCodec
import pickle

f = open('twc2016.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

# model params
weight_scale = 1e-1
layers = [80, 80, 80]

rnn = RNN(uni_chars, layers, weight_scale, False)
rnn.set_freq(100)
rnn.set_sample_len(400)
rnn.set_rollback(25)
rnn.set_padd(25)
rnn.set_clip(50)
rnn.set_codec(cc)
rnn.train_N(int_dataset, 5e4)
# rnn.set_clip(10)
# rnn.set_rollback(50)
# rnn.set_padd(50)
# rnn.set_clip(5)
# rnn.set_rollback(200)
# rnn.cont_N(int_dataset, 1e5)
# rnn.set_rollback(100)
# rnn.set_padd(100)
# rnn.cont_N(int_dataset, 1e5)
rnn.set_rollback(200)
rnn.set_padd(200)
rnn.set_sample_len(1000)
rnn.cont_NTOL(int_dataset, 5e5, 50)
f = open('rnn2016_alt.bin', 'wb')
pickle.dump(rnn, f)
f.close()
