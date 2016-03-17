#!/usr/bin/python

import numpy as np

def hrow(val, hot_len):
    onehot = np.zeros(hot_len)
    if val == -1:
        return onehot
    onehot[int(val)] = 1
    return onehot

def hcol(val, hot_len):
    onehot = np.zeros((hot_len,1))
    if val == -1:
        return onehot
    onehot[int(val)] = 1
    return onehot

def key(oh):
    val = np.argmax(oh)
    try:
        if len(val) > 1:
            raise ValueError("input not onehot!")
    except TypeError:
        0
    return val

def hcol_seq(seq, hot_len):
    seq_len = len(seq)
    onehot = np.zeros((hot_len, seq_len))
    for t in xrange(seq_len):
        val = int(seq[t])
        if val == -1:
            continue # skip this iteration
        onehot[val, t] = 1
    return onehot
