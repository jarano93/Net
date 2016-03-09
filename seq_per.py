#!usr/bin/python

from pymod.per import PerRec as PerRec
from pymod.char import CharCodec as CCodec


f = open('twcset.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

# modelparams
h_len = 120

print "%d unique characters in dataset\n\n" % uni_chars

pr = PerRec(uni_chars, h0_len, h1_len)
# rnn.set_freq(50)
pr.set_sample_len(800)
pr.set_rollback(200)
rnn.set_clip(10)
pr.set_codec(cc)
pr.train_TOL(int_dataset, 5)
pr.train_N(int_dataset, 10000)
