#!usr/bin/python

from pymod.rnn import RNN as RNN
from pymod.char import CharCodec as CCodec
import pickle

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

# model params
weight_scale = 1e-1
layers = [60, 60, 60, 60]


rnn = RNN(uni_chars, layers, weight_scale, False)
rnn.set_freq(100)
rnn.set_sample_len(400)
rnn.set_rollback(200)
rnn.set_padd(30)
rnn.set_clip(10)
rnn.set_codec(cc)
rnn.train_N(int_dataset, 1e6)
f = open('rnn4_alt.bin', 'wb')
pickle.dump(rnn, f)
f.close()
