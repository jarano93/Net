#!usr/bin/py

import numpy as np
import onehot as oh
import math

def rand_ND(shape):
    # dude, typing np.random.random_sample is not fun
    return np.random.random_sample(data_len)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - pow(np.tanh(x), 2)

def sig(x):
    if type(x) == float or type(x) == int or type(x) == np.float64:
        if x > 2e2:
            return 1
        elif x < -2e2:
            return 0
        else:
            return 1 / (1 + math.exp(-x))
    else:
        n = len(x)
        try:
            return np.ones(len(x)) / (1 + np.exp(-x))
        except FloatingPointError:
            result = np.zeros(len(x))
            for i in xrange(n):
                val = x[i]
                if val > 4e2:
                    result[i] = n
                elif val < -4e2:
                    result[i] = 0
                else:
                    result[i] = n / (1 + math.exp(-val))
            return result
    
def dsig(x):
    return sig(x) * (1 - sig(x))

def softmax(x): # x assumed to be a 1d vector
    x_exp = np.exp(x)
    denom = np.sum(np_exp)
    return x_exp / denom

# an LSTM inspired by Alex Graves' own work with RNNs of LSTMs
class LSTM:

    def __init__(self, data_len):
        self.data_len = int(data_len)

        self.data = rand_ND(data_len) # stores the most recent sequence data given to the LSTM

        self.output = rand_ND(data_len) # stores the most recent 

        self.master_step = 1e-2

        # Why aren't I defining an LSTM object?  I think bptt might be wonky between objects
        # gonna hardcode first

        ##############
        #    LSTM    #
        ##############

        # fed in the input data--a single observation in a sequence
        # its output gets fed into LSTM1

        # defines the input gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_i = np.zeros(data_len)
        self.act_i = np.zeros(data_len)
        self.w_x_i = rand_ND(data_len)
        self.w_h_i = rand_ND(data_len)
        self.w_c_i = rand_ND(data_len)
        self.b_i = rand_ND(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_i = np.zeros(data_len)
        self.hw_h_i = np.zeros(data_len)
        self.hw_c_i = np.zeros(data_len)
        self.hb_i = np.zeros(data_len)

        # defines the forget gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_f = np.zeros(data_len)
        self.act_f = np.zeros(data_len)
        self.w_x_f = rand_ND(data_len)
        self.w_h_f = rand_ND(data_len)
        self.w_c_f = rand_ND(data_len)
        self.b_f = rand_ND(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_f = np.zeros(data_len)
        self.hw_h_f = np.zeros(data_len)
        self.hw_c_f = np.zeros(data_len)
        self.hb_f = np.zeros(data_len)

        # defines the cell state, the argument of its activation function, and the biases and weights used to calculate it
        self.cell = np.zeros(data_len)
        self.act_c = np.zeros(data_len)
        self.w_x_c = rand_ND(data_len)
        self.w_h_c = rand_ND(data_len)

        # defines the former cell state
        self.cell_past = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_c = np.zeros(data_len)
        self.hw_h_c = np.zeros(data_len)

        # defines the output gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_o = np.zeros(data_len)
        self.act_o = np.zeros(data_len)
        self.w_x_o = rand_ND(data_len)
        self.w_h_o = rand_ND(data_len)
        self.w_c_o = rand_ND(data_len)
        self.b_o = rand_ND(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_o = np.zeros(data_len)
        self.hw_h_o = np.zeros(data_len)
        self.hw_c_o = np.zeros(data_len)
        self.hb_o = np.zeros(data_len)

        # the hidden output of the lstm, and the previous hidden output
        self.hidden_old = np.zeros(data_len)
        self.hidden = np.zeros(data_len)
        self.output = np.zeros(data_len)

        # yeah, im defining a modular LSTM object if I ever improve this project once I'm done
        # LSTM grids
        # LSTM cubes
        # LSTM hypercubes

    # feedforward data through the network
    def ff(self, data, verbose=True):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during feedforward")
        self.hidden_old = self.hidden
        self.cell_old = self.cell

        self.act_i = self.w_d_i * data + self.w_h_i * self.hidden + self.w_c_i * self.cell + self.b_i
        self.gate_i = sig(self.act_i)

        self.act_f = self.w_d_f * data + self.w_h_f * self.hidden + self.w_c_f * self.cell + self.b_f
        self.gate_f = sig(self.act_f)

        self.act_c = self.w_d_c * data + self.w_h_c * self.hidden + celf.b_c
        self.cell = self.gate_f * self.cell + self.gate_i * tanh(self.act_c)

        self.act_g = self.w_d_o * data + self.w_h_o * self.hidden + self.w_c_o * self.cell + self.b_o
        self.gate_o = sig(self.act_g)

        self.hidden = self.gate_o * tanh(self.cell)

        self.output = softmax(self.hidden)
        if verbose:
            return self.output

    def bptt(self, sequence):
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        seq_shape = sequence.shape
        seq_len = seq_shape[0]

        # the sequence to be fed into feedforward, starts out with all zeros
        ff_seq = np.append((np.zeros(self.data_len),  sequence), axis=0)

        # delta & gradients for the output gate
        delta_o = np.zeros(self.data_len)
        g_x_o = np.zeros(self.data_len)
        g_h_o = np.zeros(self.data_len)
        g_c_o = np.zeros(self.data_len)
        g_o = np.zeros(self.data_len)

        # delta & gradients for the cell state, and a fast link to the cell state
        delta_c = np.zeros(self.data_len)
        g_x_c = np.zeros(self.data_len)
        g_h_c = np.zeros(self.data_len)
        
        link = np.zeros(self.data_len)

        # delta & gradients for the forget gate
        delta_f = np.zeros(self.data_len)
        g_x_f = np.zeros(self.data_len)
        g_h_f = np.zeros(self.data_len)
        g_c_f = np.zeros(self.data_len)
        g_f = np.zeros(self.data_len)


        # delta & gradients for the input gate
        delta_i = np.zeros(self.data_len)
        g_x_i = np.zeros(self.data_len)
        g_h_i = np.zeros(self.data_len)
        g_c_i = np.zeros(self.data_len)
        g_i = np.zeros(self.data_len)
        
        for t in xrange(self.seq_len):
            ff_data = ff_seq[t,:]
            target = sequence[t,:]

            residue = self.ff(ff_data) - target

            delta_o = residue * tanh(self.cell) * dsig(self.act_o)
            g_x_o += delta_o * ff_data
            g_h_o += delta_o * self.hidden_old
            g_c_o += delta_o * self.cell
            g_o += delta_o

            link = residue * (tanh(self.cell) * dsig(self.act_o) * self.w_c_o + self.gate_o * dtanh(self.cell))

            delta_c = link * self.gate_i * dtanh(self.act_c)
            g_x_c += delta_c * ff_data
            g_h_c += delta_c * self.hidden_old

            delta_f = link * self.cell_old * dsig(self.act_c)
            g_x_f += delta_f * ff_data
            g_h_f += delta_f * self.hidden_old
            g_c_f += delta_f * self.cell_old
            g_f += delta_f

            delta_i = link * tanh(self.act_c) * dsig(act_i)
            g_x_i += delta_i * ff_data
            g_h_i += delta_i * self.hidden_old
            g_c_i += delta_i * self.cell_old
            g_i += delta_i

        return g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_x_o, g_h_o, g_c_o, g_o

    def sample(self, sample_len):
        result = np.zeros((self.data_len, sample_len))
        seed = np.zeros(self.data_len)
        for i in xrange(sample_len):
            result[i,:] = oh.hot(np.argmax(self.ff(seed)))
            seed = result[i,:]
        return result

    def train(self, sequence, verbose=True):
        self.bptt(sequence)
        return self.seq_loss(sequence, verbose)

    def train_N(self, sequence, N, verbose=True):
        for n in xrange(N):
            self.train(sequence, verbose)

    def train_TOL(self, sequence, TOL, verbose=True):
        while True:
            loss = self.train(sequence, verbose)
            if loss < TOL:
                break

    def seq_loss(self, sequence, verbose=True):
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        seq_shape = sequence.shape
        ff_seq = np.append(np.zeros(data_len), sequence, axis=0)
        loss = 0
        for t in xrange(len(sequence[0])):
            ff_data = ff_sequence[t,:]
            actual = sequence[t,:]
            key = int(np.argmax(actual))
            output = self.ff(ff_data)
            loss -= math.log(output[key])
        if verbose:
            print "current sequence loss: %f" % (loss)
        return loss

    def adagrad(self):
        g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_x_o, g_h_o, g_c_o, g_o = self.bptt()

        ones = np.ones(self.data_len)

        step_tol = 1e-6

        #calculate all the step sizes
        s_x_i = self.master_step / np.sqrt((step_tol * ones) + self.hw_x_i)
        s_h_i = self.master_step / np.sqrt((step_tol * ones) + self.hw_h_i)
        s_c_i = self.master_step / np.sqrt((step_tol * ones) + self.hw_c_i)
        s_i = self.master_step / np.sqrt((step_tol * ones) + self.hb_i)

        s_x_f = self.master_step / np.sqrt((step_tol * ones) + self.hw_x_f)
        s_h_f = self.master_step / np.sqrt((step_tol * ones) + self.hw_h_f)
        s_c_f = self.master_step / np.sqrt((step_tol * ones) + self.hw_c_f)
        s_f = self.master_step / np.sqrt((step_tol * ones) + self.hb_f)

        s_x_c = self.master_step / np.sqrt((step_tol * ones) + self.hw_x_c)
        s_h_c = self.master_step / np.sqrt((step_tol * ones) + self.hw_h_c)

        s_x_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_x_o)
        s_h_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_h_o)
        s_c_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_c_o)
        s_o = self.master_step / np.sqrt((step_tol * ones) + self.hb_o)

        self.w_x_i -= np.square(self.w_x_i) * g_x_i
        self.w_h_i -= np.square(self.w_h_i) * g_h_i
        self.w_c_i -= np.square(self.w_c_i) * g_c_i
        self.b_i -= np.square(self.b_i) * g_i

        self.w_x_f -= np.square(self.w_x_f) * g_x_f
        self.w_h_f -= np.square(self.w_h_f) * g_h_f
        self.w_c_f -= np.square(self.w_c_f) * g_c_f
        self.b_f -= np.square(self.b_f) * g_f

        self.w_x_c -= np.square(self.w_x_c) * g_x_c
        self.w_h_c -= np.square(self.w_h_c) * g_h_c

        self.w_x_o -= np.square(self.w_x_o) * g_x_o
        self.w_h_o -= np.square(self.w_h_o) * g_h_o
        self.w_c_o -= np.square(self.w_c_o) * g_c_o
        self.b_o -= np.square(self.b_o) * g_o

        self.archive_sq_weights()

    def archive_sq_weights(self):
        self.hw_x_i += np.square(self.w_x_i)
        self.hw_h_i += np.square(self.w_h_i)
        self.hw_c_i += np.square(self.w_c_i)
        self.hb_i += np.square(self.b_i)

        self.hw_x_f += np.square(self.w_x_f)
        self.hw_h_f += np.square(self.w_h_f)
        self.hw_c_f += np.square(self.w_c_f)
        self.hb_f += np.square(self.b_f)

        self.hw_x_c += np.square(self.w_x_c)
        self.hw_h_c += np.square(self.w_h_c)

        self.hw_x_o += np.square(self.w_x_o)
        self.hw_h_o += np.square(self.w_h_o)
        self.hw_c_o += np.square(self.w_c_o)
        self.hb_o += np.square(self.b_o)
