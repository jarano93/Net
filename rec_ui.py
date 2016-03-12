import pickle
import numpy as np

def load_rec(fName):
    bin_file = file(fName, 'rb')
    return pickle.load(bin_file)

def char_in(c, len, rec):
    if c not in rec.cc.chars:
        return rec.char_sample(np.zeros((rec.in_len, 1)), len, rec.sample)
    return rec.char_sample(rec.cc.num(c), len, rec.seed_sample)

def val_in(val, len, rec):
    if rec.text:
        return rec.char_sample(val, len, rec.seed_sample)
    return rec.seed_sample(val, len)

def save_output(fName, string):
    out = open(fName, 'a')
    out.write(string)
    out.close()
