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
        self.w_h1_data = h.rand_ND((data_len, hid1_len))
        self.w_h1_self = h.rand_ND((hid1_len, hid1_len))
        self.w_h1_prob = h.rand_ND((data_len, hid1_len))
        self.w_h1_bias = h.rand_ND(hid1_len)

        self.w_y_hid1 = h.rand_ND((hid1_len, data_len))
        self.w_y_data = h.rand_ND((data_len, data_len))
        self.w_y_self = h.rand_ND((data_len, data_len))
        self.w_y_prob = h.rand_ND((data_len, data_len))
        self.w_y_bias = h.rand_ND(data_len)

        self.zero()

    def ff(self, data):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during ff")

        for i in xrange(self.hid0_len):
            self.act_h0[i] = h.w_vdot(
                (self.w_h0_data[:,i], data),
                (self.w_h0_self[:,i], self.hid0),
                (self.w_h0_prob[:,i], self.prob)
            )
            self.act_h0[i] += self.w_h0_bias[i]
        self.hid0 = h.tanh(self.act_h0)

        for i in xrange(self.hid1_len):
            self.act_h1[i] = h.w_vdot(
                (self.w_h1_hid0[:,i], self.hid0),
                (self.w_h1_data[:,i], self.data),
                (self.w_h1_self[:,i], self.hid1),
                (self.w_h1_prob[:,i], self.prob)
            )
            self.act_h1[i] += self.w_h1_bias[i]
        self.hid1 = h.tanh(self.act_h1)

        for i in xrange(self.data_len):
            self.act_y[i] = h.w_vdot(
                (self.w_y_hid1[:,i], self.hid1),
                (self.w_y_data[:,i], self.data),
                (self.w_y_self[:,i], self.y),
                (self.w_y_prob[:,i], self.prob)
            )
            self.act_y[i] += self.w_y_bias[i]
        self.y = h.tanh(self.act_y)

        self.prob = h.softmax(self.y)
        return self.prob

    # def ff_seq
    # tbh fam I shouldn't need this if I do bptt smarter i.e. don't  cache the
    # entire activation & hidden sequences
    # just use two

    # fam... mb I need to use 3

    def bptt(self, sequence)
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        self.zero()
        seq_len = len(sequence)

        self.g_h0_data = np.zeros((data_len, hid0_len))
        self.g_h0_self = np.zeros((hid0_len, hid0_len))
        self.g_h0_prob = np.zeros((data_len, hid0_len))
        self.g_h0_bias = np.zeros(hid0_len)

        self.g_h1_hid0 = np.zeros((hid0_len, hid1_len))
        self.g_h1_data = np.zeros((data_len, hid1_len))
        self.g_h1_self = np.zeros((hid1_len, hid1_len))
        self.g_h1_prob = np.zeros((data_len, hid1_len))
        self.g_h1_bias = np.zeros(hid1_len)

        self.g_y_hid1 = np.zeros((hid1_len, data_len))
        self.g_y_data = np.zeros((data_len, data_len))
        self.g_y_self = np.zeros((data_len, data_len))
        self.g_y_prob = np.zeros((data_len, data_len))
        self.g_y_bias = np.zeros(data_len)

        seed = np.zeros(data_len)
        ff_res = np.zeros(data_len)

        hid0_old = np.zeros(self.hid0_len)
        hid1_old = np.zeros(self.hid1_len)
        y_old = np.zeros(self.data_len)
        prob_old = np.zeros(self.data_len)

        # jump values
        h0_jump = np.zeros(self.data_len)
        h1_jump = np.zeros(self.data_len)

        # fast delta values  PROTIP!  ALSO THE BIAS GRADIENTS FORALL t
        h0_delta = np.zeros(self.data_len)
        h1_delta = np.zeros(self.data_len)
        y_delta = np.zeros(self.data_len)

        for t in xrange(seq_len):
            # feedforward
            ff_res = self.ff(seed)

            # bptt here

            y_delta = (self.prob - ff_res) * h.dtanh(self.act_y)
            self.g_y_hid1 += y_delta * self.hid1
            self.g_y_data += y_delta * seed
            self.g_y_self += y_delta * y_old
            self.g_y_prob += y_delta * prob_old
            self.g_y_bias += y_delta

            h1_jump = y_delta * self.w_y_hid1
            h1_delta = h1_jump * h.dtanh(self.act_h1)
            self.g_h1_hid0 += h1_delta * self.hid0
            self.g_h1_data += h1_delta * seed
            self.g_h1_self += h1_delta * hid1_old
            self.g_h1_prob += h1_delta * prob_old
            self.g_h1_bias += h1_delta

            h0_jump = h1_delta * self.w_y_hid0
            h0_delta = h0_jump * h.dtanh(self.act_h0)
            self.g_h0_data += h0_delta * seed
            self.g_h0_self += h0_delta * hid0_old
            self.g_h0_prob += h0_delta * prob_old
            self.g_h1_bias += h0_delta

            h0_jump = h1_delta * self.w_h1_hid0
            h0_delta = h0_jump * h.dtanh(self.act_h0)

            # prep for next iteration

            act_h0_old = self.act_h0
            hid0_old = self.hid0

            act_h1_old = self.act_h1
            hid1_old = self.hid1

            act_y_old = self.act_y
            y_old = self.y

            prob_old = prob

            seed = sequence[t,:]
        # returns


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
        self.act_h0 = np.zeros(self.hid0_len)
        self.hid0 = np.zeros(self.hid0_len)

        self.act_h1 = np.zeros(self.hid1_len)
        self.hid1 = np.zeros(self.hid1_len)

        self.act_y = np.zeros(self.data_len)
        self.y = np.zeros(self.data_len)

        self.prob = np.zeros(self.data_len)
