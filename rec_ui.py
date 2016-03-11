import pickle
import numpy as np

def load_rec(fName):
    bin_file = file(fName, 'rb')
    return pickle.load(bin_file)

def char_in(c, len, rec):
    if c not in rec.cc.chars:
        return rec.sample(np.zeros((rec.in_len, 1)), len)
    return rec.seed_sample(rec.cc.num(c), len)

def val_in(val, len, rec):
    return rec.seed_sample(val, len)

def save_output(fName, string):
    out = open(fName, 'a')
    out.write(string)
    out.close()
