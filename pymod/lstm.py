#!usr/bin/py

import numpy as np
import math

#an implemention of an LSTM generative net inspired by Alex Graves's work

def sig(x):
    try:
        return np.ones(len(x)) / (1 + np.exp(x))
    except:
        return 1 / (1 + np.exp(x))
    
def dsig(x):
    return sig(x) * (1 - sig(x))

class LSTM():
    def __init__(self, in_len, out_len, mem_len, layer0_len, layer1_len):
        
    def feedforward(self, data):

    def loss(self):

    def sample(self, sample_length):

    def adagrad(self):
