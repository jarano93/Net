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

        self.w_h0_data = h.rand_ND((hid0_len, data_len))
        self.w_h0_self = h.rand_ND((hid0_len, hid0_len))
        self.w_h0_prob = h.rand_ND((hid0_len, data_len))
        self.w_h0_bias = h.rand_ND((hid0_len, 1))

        self.w_h1_hid0 = h.rand_ND((hid1_len, hid0_len))
        self.w_h1_data = h.rand_ND((hid1_len, data_len))
        self.w_h1_self = h.rand_ND((hid1_len, hid1_len))
        self.w_h1_prob = h.rand_ND((hid1_len, data_len))
        self.w_h1_bias = h.rand_ND((hid1_len, 1))

        self.w_y_hid1 = h.rand_ND((data_len, hid1_len))
        self.w_y_data = h.rand_ND((data_len, data_len))
        self.w_y_self = h.rand_ND((data_len, data_len))
        self.w_y_prob = h.rand_ND((data_len, data_len))
        self.w_y_bias = h.rand_ND((data_len, 1))

        self.zero()

    def ff(self, data):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during ff")

        for i in xrange(self.hid0_len):
            self.act_h0[i] = h.w_vdot(
                (self.w_h0_data[i], data),
                (self.w_h0_self[i], self.hid0),
                (self.w_h0_prob[i], self.prob)
            )
            self.act_h0[i] += self.w_h0_bias[i]
        self.hid0 = h.tanh(self.act_h0)

        for i in xrange(self.hid1_len):
            self.act_h1[i] = h.w_vdot(
                (self.w_h1_hid0[i], self.hid0),
                (self.w_h1_data[i], self.data),
                (self.w_h1_self[i], self.hid1),
                (self.w_h1_prob[i], self.prob)
            )
            self.act_h1[i] += self.w_h1_bias[i]
        self.hid1 = h.tanh(self.act_h1)

        for i in xrange(self.data_len):
            self.act_y[i] = h.w_vdot(
                (self.w_y_hid1[i], self.hid1),
                (self.w_y_data[i], self.data),
                (self.w_y_self[i], self.y),
                (self.w_y_prob[i], self.prob)
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

        self.g_h0_data = np.zeros((hid0_len, data_len))
        self.g_h0_self = np.zeros((hid0_len, hid0_len))
        self.g_h0_prob = np.zeros((hid0_len, data_len))
        self.g_h0_bias = np.zeros(hid0_len)

        self.g_h1_hid0 = np.zeros((hid1_len, hid0_len))
        self.g_h1_data = np.zeros((hid1_len, data_len))
        self.g_h1_self = np.zeros((hid1_len, hid1_len))
        self.g_h1_prob = np.zeros((hid1_len, data_len))
        self.g_h1_bias = np.zeros(hid1_len)

        self.g_y_hid1 = np.zeros((data_len, hid1_len))
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

            y_delta = (self.prob - sequence[t]) * h.dtanh(self.act_y)
            for i in xrange(data_len):
                for j in xrange(hid1_len):
                    self.g_y_hid1[i,j] += y_delta[i] * self.hid2[j]
                for j in xrange(data_len):
                    self.g_y_data[i,j] += y_delta[i] * seed[j]
                    self.g_y_self[i,j] += y_delta[i] * y_old[j]
                    self.g_y_prob[i,j] += y_delta[i] * prob_old[j]
            self.g_y_bias += y_delta

            for i in xrange(hid1_len):
                h1_jump[i] = h.w_vdot(y_delta, self.w_y_hid1[:,i])
            h1_delta = h1_jump * h.dtanh(self.act_h1)
            for i in xrange(hid1_len):
                for j in xrange(hid0_len):
                    self.g_h1_hid0[i] += h1_delta[i] * self.hid0[j]
                for j in xrange(data_len):
                    self.g_h1_data[i] += h1_delta[i] * seed[j]
                    self.g_h1_prob[i] += h1_delta[i] * prob_old[j]
                for j in xrange(hid1_len):
                    self.g_h1_self[i] += h1_delta[i] * hid1_old[j]
            self.g_h1_bias += h1_delta

            for i in xrange(hid0_len):
                h0_jump[i] = h.w_vdot(h1_delta, self.w_h1_hid0[:,i])
            h0_delta = h0_jump * h.dtanh(self.act_h0)
            for i in xrange(hid0_len):
                for j in xrange(data_len):
                    self.g_h0_data[i] += h0_delta[i] * seed[j]
                    self.g_h0_prob[i] += h0_delta[i] * prob_old[j]
                for j in xrange(hid0_len):
                    self.g_h0_self[i] += h0_delta[i] * hid0_old[j]
            self.g_h1_bias += h0_delta

            # prep for next iteration

            hid0_old = self.hid0
            hid1_old = self.hid1
            y_old = self.y
            prob_old = prob

            seed = sequence[t,:]

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

    # momentum must be [0,1
    def momentum_desc(self, step_size, momentum):
        self.w_h0_data -= step_size * self.g_h0_data + momentum * self.m_h0_data
        self.w_h0_self -= step_size * self.g_h0_self + momentum * self.m_h0_self
        self.w_h0_prob -= step_size * self.g_h0_prob + momentum * self.m_h0_prob
        self.w_h0_bias -= step_size * self.g_h0_bias + momentum * self.m_h0_bias

        self.w_h1_hid0 -= step_size * self.g_h1_hid0 + momentum * self.m_h1_hid0
        self.w_h1_data -= step_size * self.g_h1_data + momentum * self.m_h1_data
        self.w_h1_self -= step_size * self.g_h1_self + momentum * self.m_h1_self
        self.w_h1_prob -= step_size * self.g_h1_prob + momentum * self.m_h1_prob
        self.w_h1_bias -= step_size * self.g_h1_bias + momentum * self.m_h1_bias

        self.w_y_hid1 -= step_size *  self.g_y_hid1 + momentum * self.w_y_hid1
        self.w_y_data -= step_size *  self.g_y_data + momentum * self.w_y_data
        self.w_y_self -= step_size *  self.g_y_self + momentum * self.w_y_self
        self.w_y_prob -= step_size *  self.g_y_prob + momentum * self.w_y_prob
        self.w_y_bias -= step_size *  self.g_y_bias + momentum * self.w_y_bias

        self.m_h0_data = step_size * self.g_h0_data
        self.m_h0_self = step_size * self.g_h0_self
        self.m_h0_prob = step_size * self.g_h0_prob
        self.m_h0_bias = step_size * self.g_h0_bias

        self.m_h1_hid0 = step_size * self.g_h1_hid0
        self.m_h1_data = step_size * self.g_h1_data
        self.m_h1_self = step_size * self.g_h1_self
        self.m_h1_prob = step_size * self.g_h1_prob
        self.m_h1_bias = step_size * self.g_h1_bias

        self.m_y_hid1 =  step_size *  self.g_y_hid1
        self.m_y_data =  step_size *  self.g_y_data
        self.m_y_self =  step_size *  self.g_y_self
        self.m_y_prob =  step_size *  self.g_y_prob
        self.m_y_bias =  step_size *  self.g_y_bias

    def zero(self):
        self.act_h0 = np.zeros(self.hid0_len)
        self.hid0 = np.zeros(self.hid0_len)

        self.act_h1 = np.zeros(self.hid1_len)
        self.hid1 = np.zeros(self.hid1_len)

        self.act_y = np.zeros(self.data_len)
        self.y = np.zeros(self.data_len)

        self.prob = np.zeros(self.data_len)

    def momentum_zero(self):
        self.m_h0_data = np.zeros((hid0_len, data_len))
        self.m_h0_self = np.zeros((hid0_len, hid0_len))
        self.m_h0_prob = np.zeros((hid0_len, data_len))
        self.m_h0_bias = np.zeros((hid0_len, 1))

        self.m_h1_hid0 = np.zeros((hid1_len, hid0_len))
        self.m_h1_data = np.zeros((hid1_len, data_len))
        self.m_h1_self = np.zeros((hid1_len, hid1_len))
        self.m_h1_prob = np.zeros((hid1_len, data_len))
        self.m_h1_bias = np.zeros((hid1_len, 1))

        self.m_y_hid1 = np.zeros((data_len, hid1_len))
        self.m_y_data = np.zeros((data_len, data_len))
        self.m_y_self = np.zeros((data_len, data_len))
        self.m_y_prob = np.zeros((data_len, data_len))
        self.m_y_bias = np.zeros((data_len, 1))
