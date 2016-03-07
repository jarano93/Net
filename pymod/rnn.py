#!usr/bin/py

import numpy as np
import math as m
import onehot as oh
import help as h

#TODO debug & test

class RNN:

    def __init__(self, data_len, hid0_len, hid1_len): # OK
        self.data_len = data_len
        self.hid0_len = hid0_len
        self.hid1_len = hid1_len

        self.w_h0_data = h.rand_ND((hid0_len, data_len))
        self.w_h0_self = h.rand_ND((hid0_len, hid0_len))
        self.w_h0_prob = h.rand_ND((hid0_len, data_len))
        self.w_h0_bias = h.rand_ND(hid0_len)

        self.w_h1_hid0 = h.rand_ND((hid1_len, hid0_len))
        self.w_h1_data = h.rand_ND((hid1_len, data_len))
        self.w_h1_self = h.rand_ND((hid1_len, hid1_len))
        self.w_h1_prob = h.rand_ND((hid1_len, data_len))
        self.w_h1_bias = h.rand_ND(hid1_len)

        self.w_y_hid1 = h.rand_ND((data_len, hid1_len))
        self.w_y_data = h.rand_ND((data_len, data_len))
        self.w_y_self = h.rand_ND((data_len, data_len))
        self.w_y_prob = h.rand_ND((data_len, data_len))
        self.w_y_bias = h.rand_ND(data_len)

        self.zero()

    def ff(self, data): # OK
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during ff")

        self.act_h0 = np.zeros(self.hid0_len)
        for i in xrange(self.hid0_len):
            self.act_h0[i] = np.vdot(self.w_h0_data[i], data)
            self.act_h0[i] += np.vdot(self.w_h0_self[i], self.hid0)
            self.act_h0[i] += np.vdot(self.w_h0_prob[i], self.prob)
            self.act_h0[i] += self.w_h0_bias[i]
        self.hid0 = h.tanh(self.act_h0)

        self.act_h1 = np.zeros(self.hid1_len)
        for i in xrange(self.hid1_len):
            self.act_h1[i] = np.vdot(self.w_h1_hid0[i], self.hid0)
            self.act_h1[i] += np.vdot(self.w_h1_data[i], data)
            self.act_h1[i] += np.vdot(self.w_h1_self[i], self.hid1)
            self.act_h1[i] += np.vdot(self.w_h1_prob[i], self.prob)
            self.act_h1[i] += self.w_h1_bias[i]
        self.hid1 = h.tanh(self.act_h1)

        self.act_y = np.zeros(self.data_len)
        for i in xrange(self.data_len):
            self.act_y[i] = np.vdot(self.w_y_hid1[i], self.hid1)
            self.act_y[i] = np.vdot(self.w_y_data[i], data)
            self.act_y[i] = np.vdot(self.w_y_self[i], self.y)
            self.act_y[i] = np.vdot(self.w_y_prob[i], self.prob)
            self.act_y[i] += self.w_y_bias[i]
        self.y = h.tanh(self.act_y)

        self.prob = h.softmax(self.y)
        return self.prob

    # this ain't right. I need to use three ff caches -- DONE(?)
    def bptt(self, sequence):
        # for sequence, rows are the individual datapoints in the sequence
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        self.zero()
        seq_len = len(sequence)

        self.g_h0_data = np.zeros((self.hid0_len, self.data_len))
        self.g_h0_self = np.zeros((self.hid0_len, self.hid0_len))
        self.g_h0_prob = np.zeros((self.hid0_len, self.data_len))
        self.g_h0_bias = np.zeros(self.hid0_len)

        self.g_h1_hid0 = np.zeros((self.hid1_len, self.hid0_len))
        self.g_h1_data = np.zeros((self.hid1_len, self.data_len))
        self.g_h1_self = np.zeros((self.hid1_len, self.hid1_len))
        self.g_h1_prob = np.zeros((self.hid1_len, self.data_len))
        self.g_h1_bias = np.zeros(self.hid1_len)

        self.g_y_hid1 = np.zeros((self.data_len, self.hid1_len))
        self.g_y_data = np.zeros((self.data_len, self.data_len))
        self.g_y_self = np.zeros((self.data_len, self.data_len))
        self.g_y_prob = np.zeros((self.data_len, self.data_len))
        self.g_y_bias = np.zeros(self.data_len)

        # sequential values initialization
        # old doesn't need the activations
        hid0_old = np.zeros(self.hid0_len)
        hid1_old = np.zeros(self.hid1_len)
        y_old = np.zeros(self.data_len)
        prob_old = np.zeros(self.data_len)

        data = np.zeros(self.data_len)

        # current(cur) DOES need the activations
        ff_res_cur = self.ff(np.zeros(self.data_len))
        act_h0_cur = self.act_h0
        hid0_cur = self.hid0

        act_h1_cur = self.act_h1
        hid1_cur = self.hid1

        act_y_cur = self.act_y
        y_cur = self.y

        prob_cur = self.prob

        # gamma vals for cur
        y_gamma = (prob_cur - sequence[0])
        h1_gamma = h.vt_mult(y_gamma * h.dtanh(act_y_cur), self.w_y_hid1)
        h0_gamma = h.vt_mult(h1_gamma * h.dtanh(act_h1_cur), self.w_h1_hid0)

        # epsilon vals for future
        y_epsilon = np.zeros(self.data_len)
        h1_epsilon = np.zeros(self.hid1_len)
        h0_epsilon = np.zeros(self.hid0_len)

        # fast delta values  PROTIP!  ALSO THE BIAS GRADIENTS FORALL
        # sum together
        y_delta = np.zeros(self.data_len)
        h1_delta = np.zeros(self.hid1_len)
        h0_delta = np.zeros(self.hid0_len)

        # GET MAX LENGTHS FOR INNERMOST FOR LOOPS NOW faster this way
        max_len_y = np.amax((self.data_len, self.hid1_len))
        max_len_h1 = np.amax((self.data_len, self.hid0_len, self.hid1_len))
        max_len_h0 = np.amax((self.data_len, self.hid0_len))
        for t in xrange(seq_len - 1):
            # feedforward for the t+1 data
            self.ff(sequence[t])
            # seq_cur = sequence[t]
            seq = sequence[t+1]

            # bptt here
            y_epsilon = (self.prob - seq)
            y_delta = y_epsilon * h.vt_mult(h.dtanh(self.act_h1), self.w_y_self)
            y_delta = (y_gammma + y_delta) * h.dtanh(act_y_cur)
            for i in xrange(data_len):
                for j in xrange(max_len_y):
                    if j < hid1_len:
                        self.g_y_hid1[i,j] += y_delta[i] * self.hid1[j]
                    if j < data_len:
                        self.g_y_data[i,j] += y_delta[i] * data[j]
                        self.g_y_self[i,j] += y_delta[i] * y_old[j]
                        self.g_y_prob[i,j] += y_delta[i] * prob_old[j]
                self.g_y_bias[i] += y_delta[i]

            h1_epsilon = h.vt_mult(y_epsilon * h.dtanh(self.act_y), self.w_y_hid1)
            h1_delta = h1_epsilon * h.vt_mult(h.dtanh(self.act_h1), self.w_h1_self)
            h1_delta = (h1_gamma + h1_delta) * h.dtanh(act_h1_cur)
            for i in xrange(hid1_len):
                for j in xrange(max_len_h1):
                    if j < hid0_len:
                        self.g_h1_hid0[i] += h1_delta[i] * self.hid0[j]
                    if j < data_len:
                        self.g_h1_data[i] += h1_delta[i] * data[j]
                        self.g_h1_prob[i] += h1_delta[i] * prob_old[j]
                    if j < hid1_len:
                        self.g_h1_self[i] += h1_delta[i] * hid1_old[j]
                self.g_h1_bias[i] += h1_delta[i]

            h0_epsilon = h.vt_mult(h1_epsilon * h.dtanh(self.act_h1), self.w_h1_hid0)
            h0_delta = h0_epsilon * h.vt_mult(h.dtanh(self.act_h0), self.w_h0_self)
            h0_delta = (h1_gamma + h0_delta) * h.dtanh(act_h0_cur)
            for i in xrange(hid0_len):
                for j in xrange(max_len_h0):
                    if j < data_len:
                        self.g_h0_data[i] += h0_delta[i] * data[j]
                        self.g_h0_prob[i] += h0_delta[i] * prob_old[j]
                    if j < hid0_len:
                        self.g_h0_self[i] += h0_delta[i] * hid0_old[j]
                self.g_h0_bias[i] += h0_delta[i]

            # prep for next iteration
            y_gamma = y_epsilon
            h1_gamma = h1_epsilon
            h0_gamma = h0_epsilon

            hid0_old = hid0_cur
            hid1_old = hid1_cur
            y_old = y_cur
            prob_old = prob_cur

            act_h0_cur = self.act_h0
            hid0_cur = self.hid0

            act_h1_cur = self.act_h1
            hid1_cur = self.hid1

            act_y_cur = self.act_y
            y_cur = self.y

            prob_cur = self.prob

            data = seq[t]
            # ENDFOR

        # and then bp for T'th stuff alone
        # don't need to ff again
        y_delta = y_gamma * h.dtanh(self.act_y)
        for i in xrange(data_len):
            for j in xrange(max_len_y):
                if j < hid1_len:
                    self.g_y_hid1[i,j] += y_delta[i] * self.hid1[j]
                if j < data_len:
                    self.g_y_data[i,j] += y_delta[i] * data[j]
                    self.g_y_self[i,j] += y_delta[i] * y_old[j]
                    self.g_y_prob[i,j] += y_delta[i] * prob_old[j]
            self.g_y_bias[i] += y_delta[i]

        h1_delta = h1_gamma * h.dtanh(self.act_h1)
        for i in xrange(hid1_len):
            for j in xrange(max_len_h1):
                if j < hid0_len:
                    self.g_h1_hid0[i] += h1_delta[i] * self.hid0[j]
                if j < data_len:
                    self.g_h1_data[i] += h1_delta[i] * data[j]
                    self.g_h1_prob[i] += h1_delta[i] * prob_old[j]
                if j < hid1_len:
                    self.g_h1_self[i] += h1_delta[i] * hid1_old[j]
            self.g_h1_bias[i] += h1_delta[i]

        h0_delta = h0_gamma * h.dtanh(self.act_h0)
        for i in xrange(hid0_len):
            for j in xrange(max_len_h0):
                if j < data_len:
                    self.g_h0_data[i] += h0_delta[i] * data[j]
                    self.g_h0_prob[i] += h0_delta[i] * prob_old[j]
                if j < hid0_len:
                    self.g_h0_self[i] += h0_delta[i] * hid0_old[j]
            self.g_h0_bias[i] += h0_delta[i]

    def sample(self, sample_len): # OK
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
        self.zero()
        return loss

    def train_N(self, sequence, step_size, momentum, N, verbose=True): # OK
        self.momentum_zero()
        for i in xrange(N):
            self.bptt(sequence)
            self.momentum_descent(step_size, momentum)
            if verbose:
                print "current sequence loss: %f" % (loss)

    def train_LOSS(self, sequence, step_size, momentum, LOSS, verbose=True): # OK
        self.momentum_zero()
        loss = self.seq_loss(sequence, verbose)
        while loss > LOSS:
            self.bptt(sequence)
            self.momentum_descent(step_size, momentum)
            loss = self.seq_loss(sequence)
            if verbose:
                print "current sequence loss: %f" % (loss)

    # momentum must be [0,1)
    def momentum_desc(self, step_size, momentum): # OK
        # IDK if this is as efficient tbh fam
        self.grad_desc(step_size)

        self.w_h0_data -= momentum * self.m_h0_data
        self.w_h0_self -= momentum * self.m_h0_self
        self.w_h0_prob -= momentum * self.m_h0_prob
        self.w_h0_bias -= momentum * self.m_h0_bias

        self.w_h1_hid0 -= momentum * self.m_h1_hid0
        self.w_h1_data -= momentum * self.m_h1_data
        self.w_h1_self -= momentum * self.m_h1_self
        self.w_h1_prob -= momentum * self.m_h1_prob
        self.w_h1_bias -= momentum * self.m_h1_bias

        self.w_y_hid1 -= momentum * self.m_y_hid1
        self.w_y_data -= momentum * self.m_y_data
        self.w_y_self -= momentum * self.m_y_self
        self.w_y_prob -= momentum * self.m_y_prob
        self.w_y_bias -= momentum * self.m_y_bias

        self.cache_momentum(step_size)

    def grad_desc(self, step_size):
        self.w_h0_data -= step_size * self.g_h0_data
        self.w_h0_self -= step_size * self.g_h0_self
        self.w_h0_prob -= step_size * self.g_h0_prob
        self.w_h0_bias -= step_size * self.g_h0_bias

        self.w_h1_hid0 -= step_size * self.g_h1_hid0
        self.w_h1_data -= step_size * self.g_h1_data
        self.w_h1_self -= step_size * self.g_h1_self
        self.w_h1_prob -= step_size * self.g_h1_prob
        self.w_h1_bias -= step_size * self.g_h1_bias

        self.w_y_hid1 -= step_size * self.g_y_hid1
        self.w_y_data -= step_size * self.g_y_data
        self.w_y_self -= step_size * self.g_y_self
        self.w_y_prob -= step_size * self.g_y_prob
        self.w_y_bias -= step_size * self.g_y_bias

    def cache_momentum(self, step_size): # OK
        self.m_h0_data = step_size * self.g_h0_data
        self.m_h0_self = step_size * self.g_h0_self
        self.m_h0_prob = step_size * self.g_h0_prob
        self.m_h0_bias = step_size * self.g_h0_bias

        self.m_h1_hid0 = step_size * self.g_h1_hid0
        self.m_h1_data = step_size * self.g_h1_data
        self.m_h1_self = step_size * self.g_h1_self
        self.m_h1_prob = step_size * self.g_h1_prob
        self.m_h1_bias = step_size * self.g_h1_bias

        self.m_y_hid1 =  step_size * self.g_y_hid1
        self.m_y_data =  step_size * self.g_y_data
        self.m_y_self =  step_size * self.g_y_self
        self.m_y_prob =  step_size * self.g_y_prob
        self.m_y_bias =  step_size * self.g_y_bias

    def zero(self): # OK
        self.act_h0 = np.zeros(self.hid0_len)
        self.hid0 = np.zeros(self.hid0_len)

        self.act_h1 = np.zeros(self.hid1_len)
        self.hid1 = np.zeros(self.hid1_len)

        self.act_y = np.zeros(self.data_len)
        self.y = np.zeros(self.data_len)

        self.prob = np.zeros(self.data_len)

    def momentum_zero(self): # OK as long as init stays the same
        self.m_h0_data = np.zeros((self.hid0_len, self.data_len))
        self.m_h0_self = np.zeros((self.hid0_len, self.hid0_len))
        self.m_h0_prob = np.zeros((self.hid0_len, self.data_len))
        self.m_h0_bias = np.zeros(self.hid0_len)

        self.m_h1_hid0 = np.zeros((self.hid1_len, self.hid0_len))
        self.m_h1_data = np.zeros((self.hid1_len, self.data_len))
        self.m_h1_self = np.zeros((self.hid1_len, self.hid1_len))
        self.m_h1_prob = np.zeros((self.hid1_len, self.data_len))
        self.m_h1_bias = np.zeros(self.hid1_len)

        self.m_y_hid1 = np.zeros((self.data_len, self.hid1_len))
        self.m_y_data = np.zeros((self.data_len, self.data_len))
        self.m_y_self = np.zeros((self.data_len, self.data_len))
        self.m_y_prob = np.zeros((self.data_len, self.data_len))
        self.m_y_bias = np.zeros(self.data_len)
