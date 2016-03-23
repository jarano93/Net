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
rnn.set_rollback(50)
rnn.set_padd(3)
rnn.set_clip(10)
rnn.set_codec(cc)
rnn.train_N(int_dataset, 2e5)
rnn.set_padd(20)
rnn.set_clip(5)
rnn.cont_N(int_dataset, 3e5)
rnn.set_padd(50)
rnn.cont_N(int_dataset, 5e5)
f = open('rnn2016.bin', 'wb')
pickle.dump(rnn, f)
f.close()
