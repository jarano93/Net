#!usr/bin/python

from pymod.lstm import LSTM as LSTM
from pymod.char import CharCodec as CCodec

f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

lstm = LSTM(uni_chars)
# lstm.set_freq(50)
lstm.set_sample_len(800)
lstm.set_rollback(200)
# lstm.set_clip(10)
lstm.set_codec(cc)
lstm.train_N(int_dataset, 1000000)
