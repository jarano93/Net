#!usr/bin/python

import numpy as np
import numpy.random as nr
import onehot as oh

# TODO sig & dsig
# REAL ABASI TODO: REWRITE ALL THIS SHIT SO IT AINT SHIT

def sig(x):
    denom = 1 + np.exp(x)
    return 1 / denom

def softmax(x): # x assumed to be a 1d vector
    x_exp = np.exp(x)
    denom = np.sum(x_exp)
    return x_exp / denom

# an LSTM inspired by Alex Graves' own work with RNNs of LSTMs
class LSTM:
    """ 
        Defines a single LSTM as described by Alex Graves in the preprint of his
        book, 'Supervised Sequence Labelling with Recurrent Neural Networks' and
        his 2014 paper, 'Generating Sequences with RNN'
    """

    # yeah, im defining a modular LSTM object if I ever improve this project once I'm done
    # LSTM grids
    # LSTM cubes
    # LSTM hypercubes
    def __init__(self, x_len):
        self.x_len = int(x_len)

        # stores the most recent sequence data given to the LSTM
        self.x = -1


        # model hyperparams
        self.step_size = 1e-1
        self.clip_mag = 5
        self.rollback_len = 100

        # sample params
        self.freq = 100
        self.sample_len = 1000
        self.text = False

        # Why aren't I defining an LSTM object?  I think bptt might be wonky between objects
        # gonna hardcode first

        ##############
        #    LSTM    #
        ##############

        # fed in the input data--a single observation in a sequence
        # its output gets fed into LSTM1

        # defines the input gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_i = np.zeros(x_len)
        # self.act_i = nr.randr(data_len) # don't actually need these fam
        self.w_x_i = nr.randr(x_len)
        self.w_h_i = nr.randr(x_len)
        self.w_c_i = nr.randr(x_len)
        self.w_i = np.zeros(x_len)

        # defines the forget gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_f = np.zeros(x_len)
        # self.act_f = nr.randr(x_len)
        self.w_x_f = nr.randr(x_len)
        self.w_h_f = nr.randr(x_len)
        self.w_c_f = nr.randr(x_len)
        self.w_f = np.ones(x_len) # supposedly works better with ones

        # defines the cell state, the argument of its activation function, and the biases and weights used to calculate it
        self.cell = np.zeros(x_len)
        self.c_tan = nr.randr(x_len) # DO need this though
        self.w_x_c = nr.randr(x_len)
        self.w_h_c = nr.randr(x_len)
        self.w_c = np.zeros(x_len)

        # defines the output gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_o = np.zeros(x_len)
        # self.act_o = nr.randr(x_len)
        self.w_x_o = nr.randr(x_len)
        self.w_h_o = nr.randr(x_len)
        self.w_c_o = nr.randr(x_len)
        self.w_o = np.zeros(x_len)

        # the hidden output of the lstm
        self.hidden = np.zeros(x_len)
        self.p = np.zeros(x_len)


    # feedforward data through the network
    def ff(self, x):
        data = oh.hcol(x, self.x_len)

        i_arg = self.w_x_i * data + self.w_h_i * self.hidden + self.w_c_i * self.cell
        i_arg += self.w_i
        self.gate_i = sig(i_arg)

        f_arg = self.w_x_f * data + self.w_h_f * self.hidden + self.w_c_f * self.cell
        f_arg += self.w_f
        self.gate_f = sig(f_arg)

        c_arg = self.w_x_c * data + self.w_h_c * self.hidden + self.w_c
        self.c_tan = np.tanh(c_arg)

        self.cell = self.gate_f * self.cell + self.gate_i * self.c_tan

        o_arg = self.w_x_o * data + self.w_h_o * self.hidden + self.w_c_o * self.cell
        o_arg += self.w_o
        self.gate_o = sig(o_arg)

        self.hidden = self.gate_o * np.tanh(self.cell)

        self.p = softmax(self.hidden)
        return np.argmax(self.p)


    def bptt(self, dataseq, targets):
        seq_len = len(dataseq)

        x_seq, i_seq, f_seq, ct_seq = {}, {}, {}, {}
        c_seq, o_seq, h_seq, p_seq = {}, {}, {}, {}
        i_seq[-1] = np.copy(self.gate_i)
        f_seq[-1] = np.copy(self.gate_f)
        ct_seq[-1] = np.copy(self.c_tan)
        c_seq[-1] = np.copy(self.cell)
        o_seq[-1] = np.copy(self.gate_o)
        h_seq[-1] = np.copy(self.hidden)
        p_seq[-1] = np.copy(self.p)

        loss = 0

        for t in xrange(seq_len):
            x_seq[t] = oh.hcol(dataseq[t], self.x_len)
            self.ff(x_seq[t])
            i_seq[t] = self.gate_i
            f_seq[t] = self.gate_f
            ct_seq[t] = self.c_tan
            c_seq[t] = self.cell
            o_seq[t] = self.gate_o
            h_seq[t] = self.hidden
            p_seq[t] = self.p
            loss -= np.log(self.p[target[t]])

        # gradients for the input gate
        self.g_x_i = np.zeros(self.x_len)
        self.g_h_i = np.zeros(self.x_len)
        self.g_c_i = np.zeros(self.x_len)
        self.g_i = np.zeros(self.x_len)

        # gradients for the forget gate
        self.g_x_f = np.zeros(self.x_len)
        self.g_h_f = np.zeros(self.x_len)
        self.g_c_f = np.zeros(self.x_len)
        self.g_f = np.zeros(self.x_len)

        # gradients for the cell state, and a fast link to the cell state
        self.g_x_c = np.zeros(self.x_len)
        self.g_h_c = np.zeros(self.x_len)
        self.g_c = np.zeros(self.x_len)

        # gradients for the output gate
        self.g_x_o = np.zeros(self.x_len)
        self.g_h_o = np.zeros(self.x_len)
        self.g_c_o = np.zeros(self.x_len)
        self.g_o = np.zeros(self.x_len)

        # these are the terms used to get the partial derivatives from the t=1
        # elem as well
        c_epsilon = np.zeros(self.x_len)
        f_epsilon = np.zeros(self.x_len)
        h_epsilon = np.zeros(self.x_len)
        for t in reversed(xrange(seq_len)):
            h_delta = p_seq[t] 
            h_delta[targets[t]] -= 1
            h_delta += h_epsilon

            o_delta = h_delta * np.tanh(c_seq[t])
            o_delta = o_seq[t] * ( 1 - o_seq[t]) * o_delta
            for i in xrange(self.x_len): # faster this way
                self.g_x_o[i] += o_delta[i] * x_seq[t][i]
                self.g_h_o[i] += o_delta[i] * h_seq[t-1][i]
                self.g_c_o[i] += o_delta[i] * c_seq[t][i]
                self.g_o[i] += o_delta[i]

            c_delta = h_delta * o_seq[t] * (1 - np.square(c_seq[t]))
            c_delta += o_delta * self.w_c_o + c_epsilon
            ct_delta = c_delta * i_seq[t] * (1 - np.square(ct_seq[t])) 
            for i in xrange(self.x_len):
                self.g_x_c[i] += ct_delta[i] * x_seq[t][i]
                self.g_h_c[i] += ct_delta[i] * h_seq[t-1][i]
                self.g_c[i] += ct_delta[i]

            f_delta = c_delta + f_epsilon
            f_delta = f_seq[t] * ( 1 - f_seq[t]) * f_delta
            for i in xrange(self.x_len):
                self.g_x_f[i] += f_delta[i] * x_seq[t][i]
                self.g_h_f[i] += f_delta[i] * h_seq[t-1][i]
                self.g_c_f[i] += f_delta[i] * h_seq[t-1][i]
                self.g_f[i] += f_delta[i]

            i_delta = c_delta * ct_seq[t]
            i_delta = i_seq[t] * (1 - i_seq[t]) * i_delta
            for i in xrange(self.x_len):
                self.g_x_i[i] += i_delta[i] * x_seq[t][i]
                self.g_h_i[i] += i_delta[i] * h_seq[t-1][i]
                self.g_c_i[i] += i_delta[i] * c_seq[t-1][i]
                self.g_i[i] += i_delta[i]

            for i in xrange(self.x_len):
                c_epsilon[i] = c_delta[i] * f_seq[t][i] 
                c_epsilon[i] += f_delta[i] * self.w_c_f[i]
                f_epsilon[i] = f_delta[i] * self.w_h_f[i]
                h_epsilon[i] = o_delta[i] * self.w_h_o[i] 
                h_epsilon[i] += ct_delta[i] * self.w_h_c[i]
                h_epsilon[i] += i_delta[i] * self.w_h_i[i]

        return loss



        return loss


    def sample(self, seed, length):
        x = np.copy(seed)
        ff_seq = np.zeros(length)
        for t in xrange(length):
            ff_seq[t] = self.ff(x)
        return ff_seq

    def string_sample(self, seed, length, sample_fn):
        sample_seq = sample_fn(seed)
        return self.cc.stringify(sample_seq)

    def char_seed_sample(self, char, length):
        x = cc.int(char)
        return self.sample(x, length):

    def zero_seed_sample(self, length):
        x = np.zeros(self.x_len)
        return self.sample(x, length)

    def prep_train(self, sequence):
        prep_sequence = np.insert(sequence, 0, -1)
        seq_len = len(sequence)
        smoothloss = -np.log(1.0 / self.x_len) * seq_len
        self.reset_mem()
        self.reset_cells()
        return 0, 0, smoothloss, prep_sequence, seq_len

    def subtrain(self, n, p, smoothloss, sequence, seq_len):
        if p + self.rollback_len + 1 >= seq_len:
            p = 0
            self.reset_cells()

        if verbose and n %  self.freq == 0:
            seed = sequence[p]
            if self.text:
                print self.string_sample(seed, self.sample_len, self.sample)
                print "\n\nseed: %s, N: %d, smoothloss: %f\n\n" % (self.cc.char(seed), n, smoothloss)
            else:
                print self.sample(seed, self.sample_len)
                print "\n\nseed: %d, N: %d, smoothloss: %f\n\n" % (key, n, smoothloss)
            print "- - - - - - - - - - - - - - -\n\n"

        data_sub = sequence[p : p + self.rollback_len]
        target_sub = sequence[p + 1 : p + self.rollback_len + 1]

        loss = self.bptt(data_sub, target_sub)
        localsmooth = 0.999 * smoothloss + 0.001 * loss
        self.adagrad()

        return n + 1, p + 1, localsmooth

    def end_train(self, n, smoothloss):
        if self.text:
            print self.string_sample(1000, self.zero_seed_sample)
            print "N: %d, smoothloss: %f\n\n" % (n, smoothloss)
        else:
            print self.zero_seed_sample(1000)
            print "N: %d, smoothloss: %f\n\n" % (n, smoothloss)
        print "- - - TRAINING COMPLETE - - -"


    def train_N(self, sequence, N):
        _, p, smoothloss, prep_seq, s_len = self.prep_train(sequence)
        for n in xrange(N):
            _, p, smoothloss = self.subtrain(n, p, smoothloss, sequence, s_len)
        self.end_train(n, smoothloss)


    def train_TOL(self, sequence, TOL, verbose=True):
        n, p, smoothloss, prep_seq, s_len = self.prep_train(sequence)
        while smoothloss > TOL:
            n, p, smoothloss = self.subtrain(n, p, smoothloss, sequence, s_len)
        self.end_train(n, smoothloss)


    def adagrad(self):
        weight = [
                self.w_x_i, self.w_h_i, self.w_c_i, self.w_i,
                self.w_x_f, self.w_h_f, self.w_c_f, self.w_f,
                self.w_x_c, self.w_h_c, self.w_c,
                self.w_x_o, self.w_h_o, self.w_c_o, self.w_o
            ]
        grad = [
                self.g_x_i, self.g_h_i, self.g_c_i, self.g_i,
                self.g_x_f, self.g_h_f, self.g_c_f, self.g_f,
                self.g_x_c, self.g_h_c, self.g_c,
                self.g_x_o, self.g_h_o, self.g_c_o, self.g_o
            ]
        mem = [
                self.m_x_i, self.m_h_i, self.m_c_i, self.m_i,
                self.m_x_f, self.m_h_f, self.m_c_f, self.m_f,
                self.m_x_c, self.m_h_c, self.m_c,
                self.m_x_o, self.m_h_o, self.m_c_o, self.m_o
            ]
        for w, g, m in zip(weight, grad, mem):
            m += np.square(g)
            w -= self.step_size * g / np.sqrt(m + 1e-8)


    def reset_cells(self):
        self.gate_i = np.zeros(self.x_len)
        self.gate_f = np.zeros(self.x_len)
        self.c_tan = np.zeros(self.x_len)
        self.cell = np.zeros(self.x_len)
        self.gate_o = np.zeros(self.x_len)
        self.hidden = np.zeros(self.x_len)
        self.p = np.zeros(self.x_len)


    def reset_mem(self):
        self.m_x_i = np.zeros(self.x_len)
        self.m_h_i = np.zeros(self.x_len)
        self.m_c_i = np.zeros(self.x_len)
        self.m_i = np.zeros(self.x_len)
        self.m_x_f = np.zeros(self.x_len)
        self.m_h_f = np.zeros(self.x_len)
        self.m_c_f = np.zeros(self.x_len)
        self.m_f = np.zeros(self.x_len)
        self.m_x_c = np.zeros(self.x_len)
        self.m_h_c = np.zeros(self.x_len)
        self.m_c = np.zeros(self.x_len)
        self.m_x_o = np.zeros(self.x_len)
        self.m_h_o = np.zeros(self.x_len)
        self.m_c_o = np.zeros(self.x_len)
        self.m_o = np.zeros(self.x_len)


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
