#!/usr/bin/python

import numpy as np
import numpy.random as nr
import onehot as oh
# from char import CharCodec

# TODO make more OO

class RNN:


    def __init__(self, in_len, h0_len, h1_len, w_scale, prob=False, peek=False):
        self.prob = prob
        self.peek = peek

        self.in_len = in_len
        self.h0_len = h0_len
        self.h1_len = h1_len

        self.w_h0_x = w_scale * nr.randn(h0_len, in_len)
        self.w_h0_h0 = w_scale * nr.randn(h0_len, h0_len)
        if self.prob:
            self.w_h0_p = w_scale * nr.randn(h0_len, in_len) # prob
        self.w_h0_b = np.zeros((h0_len, 1))

        self.w_h1_h0 = w_scale * nr.randn(h1_len, h0_len)
        self.w_h1_h1 = w_scale * nr.randn(h1_len, h1_len)
        if self.peek:
            self.w_h1_x = w_scale * nr.randn(h1_len, in_len) # peek
        self.w_h1_b = np.zeros((h1_len, 1))

        self.w_y_h1 = w_scale * nr.randn(in_len, h1_len)
        self.w_y_b =  np.zeros((in_len, 1))

        self.h0 = np.zeros((self.h0_len, 1))
        self.h1 = np.zeros((self.h1_len, 1))
        self.p = np.zeros((self.in_len, 1))

        self.clip_mag = 5
        self.rollback_len = 100
        self.freq = 100
        self.sample_len = 1000
        self.step_size = 1e-1
        self.text = False


    def ff(self, data):
        # DATA MUST BE A COL VECTOR
        h0_arg = np.dot(self.w_h0_x, data)
        h0_arg += np.dot(self.w_h0_h0, self.h0) 
        if self.prob:
            h0_arg += np.dot(self.w_h0_p, self.p) # prob
        self.h0 = np.tanh(h0_arg + self.w_h0_b)

        h1_arg = np.dot(self.w_h1_h0, self.h0)
        h1_arg += np.dot(self.w_h1_h1, self.h1)
        if self.peek:
            h1_arg += np.dot(self.w_h1_x, data) # peek
        self.h1 = np.tanh(h1_arg + self.w_h1_b)

        self.y = np.dot(self.w_y_h1, self.h1) + self.w_y_b

        prob_common = np.exp(self.y)
        self.p = prob_common / np.sum(prob_common)

        return np.argmax(self.p)


    def bptt(self, dataseq, targets):
        seq_len = dataseq.shape[1]
        x_seq, h0_seq, h1_seq, y_seq, p_seq = {}, {}, {}, {}, {}
        h0_seq[-1] = np.copy(self.h0)
        h1_seq[-1] = np.copy(self.h1)
        p_seq[-1] = np.copy(self.p)
        loss = 0

        for t in xrange(seq_len):
            x_seq[t] = dataseq[:,t:t+1]
            self.ff(x_seq[t])
            key = np.argmax(targets[:,t])
            h0_seq[t] = self.h0
            h1_seq[t] = self.h1
            y_seq[t] = self.y
            p_seq[t] = self.p
            loss -= np.log(self.p[key])

        # set up grads
        self.g_h0_x = np.zeros_like(self.w_h0_x)
        self.g_h0_h0 = np.zeros_like(self.w_h0_h0)
        if self.prob:
            self.g_h0_p = np.zeros_like(self.w_h0_p) # prob
        self.g_h0_b = np.zeros_like(self.w_h0_b)

        self.g_h1_h0 = np.zeros_like(self.w_h1_h0)
        self.g_h1_h1 = np.zeros_like(self.w_h1_h1)
        if self.peek:
            self.g_h1_x = np.zeros_like(self.w_h1_x) # peek
        self.g_h1_b = np.zeros_like(self.w_h1_b)

        self.g_y_h1 = np.zeros_like(self.w_y_h1)
        self.g_y_b = np.zeros_like(self.w_y_b)

        # epsilons are for the backprop ahead time for h's
        h0_epsilon = np.zeros_like(h0_seq[0])
        h1_epsilon = np.zeros_like(h1_seq[0])

        for t in reversed(xrange(seq_len)):
            delta_y = p_seq[t] - targets[:,t].reshape(self.in_len, 1)
            self.g_y_h1 += np.dot(delta_y, h1_seq[t].T)
            self.g_y_b += delta_y

            delta_h1 = np.dot(self.w_y_h1.T, delta_y) + h1_epsilon
            delta_h1 = (1 - np.square(h1_seq[t])) * delta_h1
            self.g_h1_h0 += np.dot(delta_h1, h0_seq[t].T)
            self.g_h1_h1 += np.dot(delta_h1, h1_seq[t-1].T)
            if self.peek:
                self.g_h1_x += np.dot(delta_h1, x_seq[t].T) # peek
            self.g_h1_b += delta_h1

            delta_h0 = np.dot(self.w_h1_h0.T, delta_h1) + h0_epsilon
            delta_h0 = (1 - np.square(h0_seq[t])) * delta_h0
            self.g_h0_x += np.dot(delta_h0, x_seq[t].T)
            self.g_h0_h0 += np.dot(delta_h0, h0_seq[t-1].T)
            if self.prob:
                self.g_h0_p += np.dot(delta_h0, p_seq[t-1].T) # prob
            self.g_h0_b += delta_h0

            # prep for next iteration
            h1_epsilon = np.dot(self.w_h1_h1.T, delta_h1)
            h0_epsilon = np.dot(self.w_h0_h0.T, delta_h0)

        # clip to mitigate exploding gradients
        grad = [
                self.g_h0_x,
                self.g_h0_h0,
                self.g_h0_b,
                self.g_h1_h0,
                self.g_h1_h1,
                self.g_h1_b,
                self.g_y_h1,
                self.g_y_b
            ]
        if self.prob:
            grad += [self.g_h0_p]
        if self.peek:
            grad += [self.g_h1_x]
        for g in grad:
            np.clip(g, -self.clip_mag, self.clip_mag, out=g)

        return loss #, h0_seq[seq_len-1], h1_seq[seq_len-1] <-- in self


    def sample(self, seed, length):
        x = np.copy(seed)
        ff_seq = np.zeros(length)

        for t in xrange(length):
            key = self.ff(x)
            ff_seq[t] = key
            x = np.zeros((self.in_len, 1))
            x[key] = 1

        return ff_seq


    def char_sample(self, seed, length, sample_fn):
        sample_seq = sample_fn(seed, length)
        return self.cc.stringify(sample_seq)


    def sample_debug(self, seed, length, sample_fn):
        sample_seq = sample_fn(seed, length)
        print sample_seq
        print self.h0, self.h1, self.y, self.p
        return self.cc.stringify(sample_seq)


    def zero_sample(self, length):
        seed = np.zeros((self.in_len, 1))
        self.reset_states()
        return self.sample(seed, length) # seed is prob_prev


    def seed_sample(self, val, length):
        seed = np.zeros((self.in_len, 1))
        seed[val] = 1
        self.reset_states()
        return self.sample(seed, length)


    # train_loss & train_N take a sequence of ints, prepends a zero
    def train_TOL(self, sequence, TOL, verbose=True):
        seq_len = len(sequence)
        prep_seq = np.insert(sequence, 0, -1)
        smoothloss = - np.log(1.0 / self.in_len) * seq_len
        n, p = 0, 0
        self.reset_mem()
        self.reset_states()
        while smoothloss > TOL:
            smoothloss, n, p = self.subtrain(smoothloss, prep_seq, n, p, seq_len, verbose)


    def train_N(self, sequence, N, verbose=True):
        seq_len = len(sequence)
        prep_seq = np.insert(sequence, 0, -1)
        smoothloss = - np.log(1.0 / self.in_len) * seq_len
        p = 0
        self.reset_mem()
        self.reset_states()
        for n in xrange(N):
            smoothloss, _, p = self.subtrain(smoothloss, prep_seq, n, p, seq_len, verbose)


    def subtrain(self, smoothloss, sequence, n, p, seq_len, verbose):
        if p + self.rollback_len + 1 >= seq_len:
            p = 0
            self.reset_states()

        data_sub = oh.hcol_seq(sequence[p : p+self.rollback_len], self.in_len)
        target_sub = oh.hcol_seq(sequence[p+1 : p+self.rollback_len+1], self.in_len)

        if verbose and n % self.freq == 0:
            seed = data_sub[:,0:1]
            key = np.argmax(seed)
            if self.text:
                print self.char_sample(seed, self.sample_len, self.sample)
                # print self.sample_debug(seed, self.sample_len, self.sample)
                print "\n\nseed: %s, N: %d, smoothloss: %f\n\n" % (self.cc.char(key), n, smoothloss)
            else:
                print self.sample(seed, self.sample_len)
                print "\n\nseed: %d, N: %d, smoothloss: %f\n\n" % (key, n, smoothloss)
            print "- - - - - - - - - - - - - - -\n\n"

        loss = self.bptt(data_sub, target_sub)
        localsmooth = 0.999 * smoothloss + 0.001 * loss
        self.adagrad()

        return localsmooth, n + 1, p + 1


    def adagrad(self):
        weight = [
                self.w_h0_x, self.w_h0_h0, self.w_h0_b,
                self.w_h1_h0, self.w_h1_h1, self.w_h1_b,
                self.w_y_h1, self.w_y_b
            ]
        grad = [
                self.g_h0_x, self.g_h0_h0, self.g_h0_b,
                self.g_h1_h0, self.g_h1_h1, self.g_h1_b,
                self.g_y_h1, self.g_y_b
            ]
        mem = [
                self.m_h0_x, self.m_h0_h0, self.m_h0_b,
                self.m_h1_h0, self.m_h1_h1, self.m_h1_b,
                self.m_y_h1, self.m_y_b
            ]
        if self.prob:
            weight += [self.w_h0_p]
            grad += [self.g_h0_p]
            mem +=  [self.m_h0_p]
        if self.peek:
            weight += [self.w_h1_x]
            grad += [self.g_h1_x]
            mem +=  [self.m_h1_x]
        for w, g, m in zip(weight, grad, mem):
            m += np.square(g)
            w -= self.step_size * g / np.sqrt(m + 1e-8)


    def reset_states(self):
        self.h0 = np.zeros((self.h0_len, 1))
        self.h1 = np.zeros((self.h1_len, 1))
        self.p = np.zeros((self.in_len, 1))


    def reset_mem(self):
        self.m_h0_x = np.zeros_like(self.w_h0_x)
        self.m_h0_h0 = np.zeros_like(self.w_h0_h0)
        if self.prob:
            self.m_h0_p = np.zeros_like(self.w_h0_p) # prob
        self.m_h0_b = np.zeros_like(self.w_h0_b)

        self.m_h1_h0 = np.zeros_like(self.w_h1_h0)
        self.m_h1_h1 = np.zeros_like(self.w_h1_h1)
        if self.peek:
            self.m_h1_x = np.zeros_like(self.w_h1_x) # peek
        self.m_h1_b = np.zeros_like(self.w_h1_b)

        self.m_y_h1 = np.zeros_like(self.w_y_h1)
        self.m_y_b = np.zeros_like(self.w_y_b)


    def set_freq(self, freq):
        self.freq = freq


    def set_clip(self, val):
        self.clip_mag = val


    def set_rollback(self, val):
        self.rollback_len = val

    def set_codec(self, cc):
        self.text = True
        self.cc = cc

    def del_codec(self):
        self.text = False
        del self.cc

    def set_sample_len(self, length):
        self.sample_len = length
