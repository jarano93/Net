#!/usr/bin/py

import numpy as np

def hrow(val, hot_len):
    onehot = np.zeros(hot_len)
    onehot[int(val)] = 1
    return onehot

def hcol(val, hot_len):
    onehot = np.zeros((hot_len,1))
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
