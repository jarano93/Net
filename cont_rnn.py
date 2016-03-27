#!usr/bin/python

from pymod.char import CharCodec as CCodec
import pickle
import ui

f = open('twc2016.txt', 'r')
str_dataset = f.read().lower()
seq_length = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

print "%d unique characters in dataset\n\n" % uni_chars

rnn = ui.load_net('rnn2016TOL20.bin')
# rnn.set_clip(5)
rnn.set_sample_len(1000)
rnn.set_rollback(200)
rnn.set_padd(200)
rnn.cont_NTOL(int_dataset, 5e5, 40)
f = open('rnn2016TOL10.bin', 'wb')
pickle.dump(rnn, f)
f.close()
rnn.cont_NTOL(int_dataset, 5e5, 30)
f = open('rnn2016TOL6.bin', 'wb')
pickle.dump(rnn, f)
f.close()
rnn.cont_NTOL(int_dataset, 5e5, 20)
f = open('rnn2016TOL4.bin', 'wb')
pickle.dump(rnn, f)
f.close()
