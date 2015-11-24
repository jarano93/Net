#!/usr/bin/py

import numpy as np

def hot(val, hot_len):
    onehot = np.zeros(hot_len)
    onehot[int(val)] = 1
    return onehot

def key(oh):
    val = np.nonzero(a)
    if len(val) > 1:
        raise ValueError("input not onehot!")
    return val
