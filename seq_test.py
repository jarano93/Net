#!usr/bin/py

import numpy as np
import math as m
import onehot as oh
from char import CharCodec as CCodec
import help as h

# LITTLE SPEEDY SCRIPT
# more like a proof of concept & application of said concept
#we cross entropy now fam

# text input
f = open('twcshort.txt', 'r')
str_dataset = f.read().lower()
dataset_len = len(str_dataset)
cc = CCodec(str_dataset)
int_dataset = cc.sequence()
uni_chars = cc.length()

dataset = np.zeros((uni_chars, datset_len + 1))
for t in xrange(dataset_len):
    dataset[:,t+1] = oh.hcol(int_dataset[t], uni_chars) #dataset zero prepended

# sample settings
sample_freq = 100
sample_len = 300
supersample_len = 10000

# HYPERPARAMETERS
rollback_len = 25
hid_len = 100
step_size = 1e-1
clip_mag = 10

# model params
W_x_h = 0.01 * h.randn_ND(hid_len, uni_chars)
W_h_h = 0.01 * h.randn_ND(hid_len, hid_len)
W_h_y = 0.01 * h.randn_ND(uni_chars, hid_len)
b_h = h.randn_ND((hid_len, 1))
b_y = h.randn_ND((num_chars, 1))

def ff(data, h_state):
    '''
        feedforward single datapoint through the net
        returns hidden value, output, and softmaxed output (probability)
        data is num_charx x 1 array of input data
        h_state is the hid_len x 1 array of the hidden state
    '''
    hid = np.tanh(np.dot(W_x_h, data) + np.dot(W_h_h, h_state) + b_h)

    y = np.dot(W_h_y, hid) + b_y
    prob = h.softmax(y) # only want prob since using cross entropy loss
    return hid, y, prob # don't return h_act b/c using tanh derivative

# remove this
def adagrad(param, grad, mem):
    '''
        performs adagrad to update the net
        after update caches grad in mem
        returns mem
    '''
    pass

def train(dataseq, targets, h_state, clip_val):
    '''
        trains a net on a subset of the entire datasequence
        returns the loss, gradients of the params, and the last hidden state
        dataseq is the data sequence to train on
        target is the target sequence for the net
        h_state is the start state of the hidden layer
        USE CROSS ENTROPY FOR LOSS
    '''
    seq_len = len(dataseq)
    # I don't think I need x_seq
    # REMEMBER, WE'RE USING DICTS NOW
    h_seq, y_seq, p_seq = {}, {}, {}
    h_seq[-1] = np.copy(h_state)
    loss = 0
    for t in xrange(seq_len):
        x_seq[t] = dataseq[t]
        h_seq[t], y_seq[t], p_seq[t] = ff(dataseq[t], h_seq[t-1])
        loss += - m.log(p_seq[t][np.argmax(targets[:,t])])

    G_x_h = np.zeros_like(W_x_h)
    G_h_h = np.zeros_like(w_h_h)
    G_h_y = np.zeros_like(W_h_y)
    g_h = np.zeros_like(b_h)
    g_y = np.zeros_like(b_y)
    h_epsilon = np.zeros_like(h_seq[0])

    for t in reversed(xrange(seq_len)):
        delta_y = p_seq[t] - targets[:,t]
        G_h_y += np.dot(delta_y, h_seq[:,t].T)
        g_y += delta_y

        delta_h = np.dot(W_h_y.T, delta_y) + h_epsilon
        delta_h = (1 - h_seq[t] * h_seq[t]) * delta_h # dtanh = 1 - tanh^2
        G_x_h += np.dot(delta_h, dataseq[:,t].T)
        G_h_h += np.dot(delta_h, h_seq[:,t-1].T)
        g_h += delta_h
        h_epsilon = np.dot(W_h_h.T, delta_h)

    # clip to limit exploding grad problem
    for grad in [G_x_h, G_h_h, G_h_y, g_h, g_y]:
        np.clip(grad, -clip_val, clip_val, out=dparam)

    return loss, G_x_h, G_h_h, G_h_y, g_h, g_y, h_seq[seq_len - 1]

def sample(seed, len, cc_obj, verbose=True):
    '''
        sample makes the model generate a string of chars
        len is the length of the sample
        cc_obj is the char codec to be used for generating chars
    '''
    h_state = np.copy(seed)
    max_seq = np.zeros(len)
    prob = []
    for t in xrange(len):
        h_state, _, prob = ff(seed, h_state)
        seed = np.argmax(prob)
        max_seq[t] = seed
    del h_state, _

    sample_string = cc_obj.string(max_seq)
    if verbose:
        print sample_string
    return sample_string

def supersample(len, cc_obj, verbose=True):
    '''
        supersample calls sample with the zero seed
    '''
    zero_seed = np.zeros((uni_chars, 1))
    return sample(zero_seed, len, cc_obj, verbose)

# iteration number & index pointer
n, p = 0, 0

# set up vars for adagrad
M_x_h = np.zeros_like(W_x_h)
M_h_h = np.zeros_like(M_h_h)
M_h_y = np.zeros_like(M_h_y)
m_h = np.zeros_like(m_h)
m_y = np.zeros_like(m_y)

smooth_loss = - m.log(1.0/uni_chars) * dataset_len
loss = 0
h_state = np.zeros((hid_len, 1))
while True:
    if p + seq_len + 1 >= dataset_len:
        p = 0
        h_state = np.zeros((hid_len, 1))

    data_sub = dataset[:, p : p+seq_len]
    target_sub = dataset[:, p+1 : p+seq_len+1]

    if n % sample_freq == 0:
        print "N = %d, smoothloss = %f\n\nSAMPLE:\n" % (n, smooth_loss)
        sample(data_subset[:,0], sample_len, cc) 
        print "\n\n"

    loss, G_x_h, G_h_h, G_h_y, g_h, g_y, h_state = train(data_sub, target_sub, h_state, clip_mag)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # perform adagrad
    for param, grad, mem in zip(
        [W_x_h, W_h_h, W_h_y, w_h, w_y],
        [G_x_h, G_h_h, G_h_y, g_h, g_y],
        [M_x_h, M_h_h, M_h_y, m_h, m_y]
    ):
        param -= step_size * grad / np.sqrt(mem + 1e-8)
        mem += grad * grad


    p += seq_len
    n += 1
