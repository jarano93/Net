#!usr/bin/py

import numpy as np
import math as m
import onehot as oh
import help as h


class RNN:

    def __init__(self, data_len, hid0_len, hid1_len):
        self.data_len = data_len
        self.hid0_len = hid0_len
        self.hid1_len = hid1_len

        self.w_h0_data = h.rand_ND((data_len, hid0_len))
        self.w_h0_self = h.rand_ND((hid0_len, hid0_len))
        self.w_h0_prob = h.rand_ND((data_len, hid0_len))
        self.w_h0_bias = h.rand_ND(hid0_len)

        self.w_h1_hid0 = h.rand_ND((hid0_len, hid1_len))
        self.w_h1_peek = h.rand_ND((data_len, hid1_len))
        self.w_h1_self = h.rand_ND((hid1_len, hid1_len))
        self.w_h1_prob = h.rand_ND((data_len, hid1_len))
        self.w_h1_bias = h.rand_ND(hid1_len)

        self.w_y_hid1 = h.rand_ND((hid1_len, data_len))
        self.w_y_peek = h.rand_ND((data_len, data_len))
        self.w_y_self = h.rand_ND((data_len, data_len))
        self.w_y_prob = h.rand_ND((data_len, data_len))
        self.w_y_bias = h.rand_ND((data_len))

        self.zero()

    def ff(self, data):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during ff")

        act_h0 = np.zeros(self.hid0_len)
        for i in xrange(self.hid0_len):
            act_h0[i] = h.w_vdot(
                (self.w_h0_data[:,i], data),
                (self.w_h0_self[:,i], self.hid0),
                (self.w_h0_prob[:,i], self.prob)
            )
            act_h0[i] += self.w_h0_bias[i]
        self.hid0 = h.tanh(act_h0)

        act_h1 = np.zeros(self.hid1_len)
        for i in xrange(self.hid1_len):
            act_h1[i] = h.w_vdot(
                (self.w_h1_hid0[:,i], self.hid0),
                (self.w_h1_peek[:,i], self.data),
                (self.w_h1_self[:,i], self.hid1),
                (self.w_h1_prob[:,i], self.prob)
            )
            act_h1[i] += self.w_h1_bias[i]
        self.hid1 = h.tanh(act_h1)

        act_y = np.zeros(self.data_len)
        for i in xrange(self.data_len):
            act_y[i] = h.w_vdot(
                (self.w_y_hid1[:,i], self.hid1),
                (self.w_y_peek[:,i], self.data),
                (self.w_y_self[:,i], self.y),
                (self.w_y_prob[:,i], self.prob)
            )
            act_y[i] += self.w_y_bias[i]
        self.y = h.tanh(act_y)

        self.prob = h.softmax(self.y)
        return self.prob

    # def ff_seq

    # def bptt

    def sample(self, sample_len):
        self.zero()
        gen_sample = np.zeros((sample_len, self.data_len))
        seed = np.zeros(self.data_len)
        for t in xrange(sample_len):
            gen_sample[t,:] = oh.hot(np.argmax(self.ff(seed)), self.data_len)
            seed = gen_sample[t,:]
        return gen_sample

    def seq_loss(self, sequence, verbose=True):
        if len(sequence[0]) != self.data_len:
            raise("Unexpected data length during seq_loss")
        self.zero()
        seed = np.zeros(self.data_len)
        loss = 0
        for t in xrange(len(sequence)):
            actual_data = sequence[t,:]
            key = int(np.argmax(actual_data))
            output = self.ff(seed)
            loss -= m.log(output[key])
            seed = sequence[t,:]
        if verbose:
            print "current sequence loss: %f" % (loss)
        return loss


    # def train

    # def train_N

    # def train_LOSS

    # def momentum_descent

    def zero(self):
        self.hid0 = np.zeros(self.hid0_len)
        self.hid1 = np.zeros(self.hid1_len)
        self.y = np.zeros(self.data_len)
        self.prob = np.zeros(self.data_len)
