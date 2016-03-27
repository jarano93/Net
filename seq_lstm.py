#!usr/bin/python

from pymod.lstm2 import LSTM as LSTM
from pymod.char import CharCodec as CCodec

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

weight_scale = 1e-1
hid_len = 100

lstm = LSTM(uni_chars, hid_len, uni_chars, weight_scale)
lstm.set_freq(50)
lstm.set_sample_len(100)
lstm.set_rollback(20)
lstm.set_clip(5)
lstm.set_codec(cc)
lstm.train_N(int_dataset, 1000000)
