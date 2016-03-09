#!usr/bin/python

import numpy as np
import numpy.random as nr
import onehot as oh

def PerRec:


    # good
    def __init__(self, x_len, h_len):
        self.x_len = x_len
        self.h_len = h_len

        self.w_h_x = 0.1 * randn(h_len, x_len)
        self.w_h_h = 0.1 * randn(h_len, h_len)
        self.w_h_p = 0.1 * randn(h_len, x_len) # prob
        self.w_h_b = np.zeros((h_len, 1))

        self.w_y_h = 0.01 * randn(x_len, h_len)
        self.w_y_b =  np.zeros((x_len, 1))

        self.h = np.zeros((h_len, 1))
        self.p = np.zeros((x_len, 1))


        self.clip_mag = 10
        self.rollback_len = 100
        self.freq = 100
        self.sample_len = 1000
        self.step_size = 1e-1


    # good
    def ff(self, data):
        # DATA MUST BE A COL VECTOR
        h_arg = np.dot(self.w_h_x, data)
        h_arg += np.dot(self.w_h_h0, self.h) 
        h_arg += np.dot(self.w_h_p, self.p) # prob
        self.h = np.tanh(h_arg + self.w_h_b)

        self.y = np.dot(self.w_y_h, self.h) + self.w_y_b

        prob_common = np.exp(self.y)
        self.p = prob_common / np.sum(prob_common)

        return np.argmax(self.p)


    # OK
    def bptt(self, dataseq, targets):
        seq_len = dataseq.shape[1]
        x_seq, h_seq, y_seq, p_seq = {}, {}, {}, {}
        h_seq[-1] = np.copy(self.h)
        p_seq[-1] = np.copy(self.p)
        loss = 0

        for t in xrange(seq_len):
            x_seq[t] = dataseq[:,t:t+1]
            key = self.ff(x_seq[t])
            h_seq[t] = self.h
            y_seq[t] = self.y
            p_seq[t] = self.p
            loss -= np.log(self.p[key])

        # set up grads
        self.g_h_x = np.zeros_like(self.w_h_x)
        self.g_h_h = np.zeros_like(self.w_h_h)
        self.g_h_p = np.zeros_like(self.w_h_p) # prob
        self.g_h_b = np.zeros_like(self.w_h_b)

        self.g_y_h = np.zeros_like(self.w_y_h)
        self.g_y_b = np.zeros_like(self.w_y_b)

        # epsilons are for the backprop ahead time for h's
        h_epsilon = np.zeros_like(h_seq[0])

        for t in reversed(xrange(seq_len)):
            delta_y = p_seq[t] - targets[:,t].reshape(self.x_len, 1)
            self.g_y_h += np.dot(delta_y, h_seq[t].T)
            self.g_y_b += delta_y

            delta_h = np.dot(self.w_y_h.T, delta_y) + h_epsilon
            delta_h = (1 - np.square(h_seq[t])) * delta_h
            self.g_h_x += np.dot(delta_h, x_seq[t].T)
            self.g_h_h += np.dot(delta_h, h_seq[t].T)
            self.g_h_p += np.dot(delta_h, p_seq[t].T) # prob
            self.g_h_b += delta_h

            # prep for next iteration
            h_epsilon = np.dot(self.w_h_h.T, delta_h)

        # clip to mitigate exploding gradients
        for grad in [
                self.g_h_x,
                self.g_h_h,
                # self.g_h_p, # prob
                self.g_h_b,
                self.g_y_h,
                self.g_y_b
            ]:
            np.clip(grad, -self.clip_mag, self.clip_mag, out=grad)

        return loss #, h0_seq[seq_len-1], h1_seq[seq_len-1] <-- in self


    def sample(self, seed, length):
        x = np.copy(seed)
        ff_seq = np.zeros(length)

        for t in xrange(length):
            key = self.ff(x)
            ff_seq[t] = key
            x = np.zeros((self.x_len, 1))
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
        seed = np.zeros((self.x_len, 1))
        self.reset_states()
        return self.sample(seed, length) # seed is prob_prev


    def seed_sample(self, val, length):
        seed = np.zeros((self.x_len, 1))
        seed[val] = 1
        self.reset_states()
        return self.sample(seed, length)


    # train_loss & train_N take a sequence of ints, prepends a zero
    def train_TOL(self, sequence, TOL, verbose=True):
        seq_len = len(sequence)
        prep_seq = np.insert(sequence, 0, -1)
        smoothloss = - np.log(1.0 / self.x_len) * seq_len
        n, p = 0, 0
        self.reset_mem()
        self.reset_states()
        while smoothloss > TOL:
            smoothloss, n, p = self.subtrain(smoothloss, prep_seq, n, p, seq_len, verbose)


    def train_N(self, sequence, N, verbose=True):
        seq_len = len(sequence)
        prep_seq = np.insert(sequence, 0, -1)
        smoothloss = - np.log(1.0 / self.x_len) * seq_len
        p = 0
        self.reset_mem()
        self.reset_states()
        for n in xrange(N):
            smoothloss, _, p = self.subtrain(smoothloss, prep_seq, n, p, seq_len, verbose)


    def subtrain(self, smoothloss, sequence, n, p, seq_len, verbose):
        if p + self.rollback_len + 1 >= seq_len:
            p = 0
            self.reset_states()

        data_sub = oh.hcol_seq(sequence[p : p+self.rollback_len], self.x_len)
        target_sub = oh.hcol_seq(sequence[p+1 : p+self.rollback_len+1], self.x_len)

        if verbose and n % self.freq == 0:
            seed = data_sub[:,0:1]
            if self.text:
                print self.char_sample(seed, self.sample_len, self.sample)
                # print self.sample_debug(seed, self.sample_len, self.sample)
            else:
                print self.sample(seed, self.sample_len)
            print "\n\nN: %d, smoothloss: %f\n\n" % (n, smoothloss)
            print "- - - - - - - - - - - - - - -\n\n"

        loss = self.bptt(data_sub, target_sub)
        localsmooth = 0.999 * smoothloss + 0.001 * loss
        self.adagrad()

        return localsmooth, n + 1, p + 1


    def adagrad(self):
        for weight, grad, mem in zip(
            [
                self.w_h_x, self.w_h_h, self.w_h_b, self.w_h_p,
                self.w_y_h, self.w_y_b
            ],
            [
                self.g_h_x, self.g_h_h, self.g_h_b, self.g_h_p, 
                self.g_y_h1, self.g_y_b,
            ],
            [
                self.m_h_x, self.m_h_h, self.m_h_b, self.m_h_p, 
                self.m_y_h, self.m_y_b
            ]
        ):
            mem += np.square(grad)
            weight -= self.step_size * grad / np.sqrt(mem + 1e-8)
            # perturb the weights
            # weight *= (1.0 + 0.01 * nr.randn(weight.shape[0], weight.shape[1]))


    def reset_states(self):
        self.h = np.zeros((self.h_len, 1))
        self.p = np.zeros((self.x_len, 1))


    def reset_mem(self):
        self.m_h_x = np.zeros_like(self.w_h_x)
        self.m_h_h = np.zeros_like(self.w_h_h0)
        self.m_h_p = np.zeros_like(self.w_h_p) # prob
        self.m_h_b = np.zeros_like(self.w_h_b)

        self.m_y_h = np.zeros_like(self.w_y_h)
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
