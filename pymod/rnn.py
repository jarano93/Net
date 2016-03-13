#!usr/bin/py

import numpy as np
import numpy.random as nr
import onehot as oh
from perceptron import Perceptron


class RNN:

    def __init__(self, x_len, h_lens, w_scale, peek=True):
        self.x_len = x_len
        self.h_num = len(h_lens)
        self.peek = peek

        # set up perceptron layers

        # hyperparams
        self.clip_mag = 5
        self.rollback = 100
        self.freq = 100
        self.step_size = 1e-1
        self.padd = self.rollback

        # sample params
        self.sample_len = 1000
        self.text = False

        self.h= {}
        self.h[0] = Perceptron(x_len, x_len, h_lens[0], w_scale, False)
        for i in xrange(1, self.h_num):
            if peek:
                self.h[i] = Perceptron(x_len, h_lens[i-1], h_lens[i], w_scale, True)
            else:
                self.h[i] = Perceptron(x_len, h_lens[i-1], h_lens[i], w_scale, False)

        self.wi = w_scale * nr.randn(x_len, h_lens[-1])
        self.wb = w_scale * nr.randn(x_len, 1)
        self.y = np.zeros((x_len, 1))
        self.p = np.zeros((x_len, 1))

        self.grad_reset()
        self.mem_reset()


    def ff(self, data): # OK
        self.x = oh.hcol(data, self.x_len)
        res = self.h[0].ff(self.x)
        for i in xrange(1, self.h_num):
            if self.peek:
                res = self.h[i].ff(res, self.x)
            else:
                res = self.h[i].ff(res)
        self.y = np.dot(self.wi, res) + self.wb
        p_common = np.exp(self.y)
        self.p = p_common / np.sum(p_common)
        return np.argmax(self.p)


    def bptt(self, dataseq, targets):
        seq_len = len(dataseq)
        x_seq, h_seq, y_seq, p_seq = {}, {}, {}, {}
        for i in xrange(self.h_num):
            temp = {}
            temp[-1] = self.h[i].h
            h_seq[i] = temp
        loss = 0
        for t in xrange(seq_len):
            self.ff(dataseq[t])
            x_seq[t] = self.x
            y_seq[t] = self.y
            p_seq[t] = self.p
            for i in xrange(self.h_num):
                h_seq[i][t] = self.h[i].h
            loss -= np.log(self.p[targets[t]])
        # zero grads in init and after each adagrad
        epsilon = {}
        for i in xrange(self.h_num):
            epsilon[i] = np.zeros_like(self.h[i].wb)
        for t in reversed(xrange(seq_len)):
            delta = p_seq[t]
            delta[targets[t]] -= 1
            self.gi += np.dot(delta, h_seq[self.h_num-1][t].T)
            self.gb += delta
            delta = np.dot(self.wi.T, delta)
            for i in reversed(xrange(1, self.h_num)):
                # print str(i) + ': ' + str(h_seq[i][t].shape)
                if self.h[i].peek:
                    delta, epsilon[i] = self.h[i].bp(delta, epsilon[i], h_seq[i-1][t], h_seq[i][t], h_seq[i][t-1], x_seq[t])
                else:
                    delta, epsilon[i] = self.h[i].bp(delta, epsilon[i], h_seq[i-1][t], h_seq[i][t], h_seq[i][t-1])
            delta, epsilon[0] = self.h[0].bp(delta, epsilon[0], x_seq[t], h_seq[0][t], h_seq[0][t-1])
        self.clip_grads()
        return loss


    def clip_grads(self):
        for g in [self.gi, self.gb]:
            np.clip(g, -self.clip_mag, self.clip_mag, out=g)
        for i in xrange(self.h_num):
            self.h[i].clip_grads()


    def sample(self, seed, sample_len):
        x = np.copy(seed)
        ff_seq = np.zeros(sample_len)
        for t in xrange(sample_len):
            x = self.ff(x)
            ff_seq[t] = x
        if self.text:
            return self.cc.stringify(ff_seq)
        else:
            return ff_seq


    def clean_sample(self, sample_len):
        self.reset()
        return self.sample(-1, sample_len)


    def ui_sample(self, sample_len):
        ui_seed = raw_input('Enter a single character or an valid int:\n\t>')
        seed = -1
        if self.text:
            c_seed = ui_seed[0]
            if c_seed in self.cc.chars:
                seed = self.cc.num(c_seed[0])
        else:
            v_seed = int(ui_seed)
            if -1 < v_seed and v_seed < self.x_len:
                seed = int(char_seed)
        return self.sample(seed, sample_len)


    def prep_train(self, sequence):
        prep_sequence = np.insert(sequence, 0, -1)
        seq_len = len(sequence)
        smoothloss = -np.log(1.0 / self.x_len) * seq_len
        self.reset()
        self.mem_reset()
        return 0, 0, prep_sequence, smoothloss, seq_len

    def subtrain(self, n, p, sequence, smoothloss, seq_len):
        if p + self.rollback + 1 >= seq_len:
            p = 0
            self.reset()
        if n %  self.freq == 0:
            states = self.get_states()
            seed = sequence[p]
            print self.sample(seed, self.sample_len)
            if self.text:
                print "\n\nseed: %s, N: %d, smoothloss: %f\n\n" % (self.cc.char(seed), n, smoothloss)
            else:
                print "\n\nseed: %d, N: %d, smoothloss: %f\n\n" % (seed, n, smoothloss)
            print "- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n"
            self.set_states(states)
        data_sub = sequence[p : p + self.rollback]
        target_sub = sequence[p + 1 : p + self.rollback + 1]
        loss = self.bptt(data_sub, target_sub)
        localsmooth = 0.999 * smoothloss + 0.001 * loss
        self.adagrad()

        return n + 1, p + self.padd, localsmooth

    def end_train(self, n, smoothloss):
        print self.sample(-1, self.sample_len)
        print "N: %d, smoothloss: %f\n\n" % (n, smoothloss)
        print "*****\tTRAINING COMPLETE\t*****"

    def train_N(self, sequence, N):
        _, p, prep_seq, smoothloss, s_len = self.prep_train(sequence)
        for n in xrange(int(N)):
            _, p, smoothloss = self.subtrain(n, p, prep_seq, smoothloss, s_len)
        self.end_train(n, smoothloss)


    def train_TOL(self, sequence, TOL):
        n, p, prep_seq, smoothloss, s_len = self.prep_train(sequence)
        while smoothloss > TOL:
            n, p, smoothloss = self.subtrain(n, p, prep_seq, smoothloss, s_len)
        self.end_train(n, smoothloss)

    def adagrad(self):
        for w, g, m in zip(
                [self.wi, self.wb], [self.gi, self.gb], [self.mi, self.mb]
            ):
            m += np.square(g)
            w -= self.step_size * g / np.sqrt(m + 1e-8)
        for i in xrange(self.h_num):
            self.h[i].adagrad(self.step_size)
        self.grad_reset()

    def get_states(self):
        states = {}
        for i in xrange(self.h_num):
            states[i] = self.h[i].h
        return states

    def set_states(self, states):
        for i in xrange(self.h_num):
            self.h[i].h = states[i]

    def reset(self):
        self.y = np.zeros_like(self.y)
        self.p = np.zeros_like(self.p)
        for i in xrange(self.h_num):
            self.h[i].reset()

    def grad_reset(self):
        self.gi = np.zeros_like(self.wi)
        self.gb = np.zeros_like(self.wb)
        for i in xrange(self.h_num):
            self.h[i].grad_reset()

    def mem_reset(self):
        self.mi = np.zeros_like(self.wi)
        self.mb = np.zeros_like(self.wb) 
        for i in xrange(self.h_num):
            self.h[i].mem_reset()


    def set_freq(self, freq):
        self.freq = freq


    def set_clip(self, val):
        self.clip_mag = val
        for i in xrange(self.h_num):
            self.h[i].set_clip(val)


    def set_rollback(self, val):
        self.rollback= val

    def set_codec(self, cc):
        self.text = True
        self.cc = cc

    def del_codec(self):
        self.text = False
        del self.cc

    def set_sample_len(self, length):
        self.sample_len = length

    def set_padd(self, val):
        self.padd = val
