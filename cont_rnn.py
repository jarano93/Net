#!usr/bin/python

from pymod.char import CharCodec as CCodec
import pickle
import ui

f = open('twc_clean.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

rnn = ui.load_net('rnn4.bin')
rnn.set_clip(5)
rnn.set_padd(40)
rnn.cont_N(int_dataset, 5e5)
f = open('rnn4_cont.bin', 'wb')
pickle.dump(rnn, f)
f.close()
