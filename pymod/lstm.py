#!usr/bin/py

import numpy as np
import onehot as oh
import math

def rand_ND(shape):
    # dude, typing np.random.random_sample is not fun
    rand = np.random.random_sample(shape) - 1 
    return rand


def tanh(x):
    return np.tanh(x)

def dtanh(x):
    try:
        return 1 - np.square(tanh(x))
    except FloatingPointError:
        res = np.zeros(len(x))
        for i in xrange(len(x)):
            arg = tanh(x[i])
            if abs(arg) < 1e-4:
                arg = 0
            res[i] = 1 - np.square(arg)
        return res
            

def sig(x):
    np.seterr(over='raise', under='raise')
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
        except:
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
        self.hw_x_i = np.square(self.w_x_i)
        self.hw_h_i = np.square(self.w_h_i)
        self.hw_c_i = np.square(self.w_c_i)
        self.hb_i = np.square(self.b_i)

        # defines the forget gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_f = np.zeros(data_len)
        self.act_f = np.zeros(data_len)
        self.w_x_f = rand_ND(data_len)
        self.w_h_f = rand_ND(data_len)
        self.w_c_f = rand_ND(data_len)
        self.b_f = rand_ND(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_f = np.square(self.w_x_f)
        self.hw_h_f = np.square(self.w_h_f)
        self.hw_c_f = np.square(self.w_c_f)
        self.hb_f = np.square(self.b_f)

        # defines the cell state, the argument of its activation function, and the biases and weights used to calculate it
        self.cell = np.zeros(data_len)
        self.act_c = np.zeros(data_len)
        self.w_x_c = rand_ND(data_len)
        self.w_h_c = rand_ND(data_len)
        self.b_c = rand_ND(data_len)

        # defines the former cell state
        # self.cell_past = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_c = np.square(self.w_x_c)
        self.hw_h_c = np.square(self.w_h_c)
        self.hw_c = np.square(self.b_c)

        # defines the output gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_o = np.zeros(data_len)
        self.act_o = np.zeros(data_len)
        self.w_x_o = rand_ND(data_len)
        self.w_h_o = rand_ND(data_len)
        self.w_c_o = rand_ND(data_len)
        self.b_o = rand_ND(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_o = np.square(self.w_x_o)
        self.hw_h_o = np.square(self.w_h_o)
        self.hw_c_o = np.square(self.w_c_o)
        self.hb_o = np.square(self.b_o)

        # the hidden output of the lstm, and the previous hidden output
        # self.hidden_old = np.zeros(data_len)
        self.hidden = np.zeros(data_len)
        self.output = np.zeros(data_len)

    # feedforward data through the network
    def ff(self, data, verbose=True):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during feedforward")
        self.hidden_old = self.hidden
        self.cell_old = self.cell

        self.act_i = self.w_x_i * data + self.w_h_i * self.hidden + self.w_c_i * self.cell + self.b_i
        self.gate_i = sig(self.act_i)

        self.act_f = self.w_x_f * data + self.w_h_f * self.hidden + self.w_c_f * self.cell + self.b_f
        self.gate_f = sig(self.act_f)

        self.act_c = self.w_x_c * data + self.w_h_c * self.hidden + self.b_c
        self.cell = self.gate_f * self.cell + self.gate_i * tanh(self.act_c)

        self.act_o = self.w_x_o * data + self.w_h_o * self.hidden + self.w_c_o * self.cell + self.b_o
        self.gate_o = sig(self.act_o)

        self.hidden = self.gate_o * tanh(self.cell)

        self.output = softmax(self.hidden)
        if verbose:
            return self.output

    def ff_seq(self, sequence):
        if sequence.shape[1] != self.data_len:
            raise ValueError("Unexpected data length during sequence feedforward")
        ff_sequence = np.vstack((np.zeros(self.data_len), sequence)
        seq_shape = sequence.shape
        pad_shape = (seq_shape[0] + 1, seq_shape[1])
        seq_len = sequence.shape[0]

        act_i_seq = np.zeros(seq_shape)
        gate_i_seq = np.zeros(seq_shape)
        act_f_seq = np.zeros(seq_shape)
        gate_f_seq = np.zeros(seq_shape)
        act_c_seq = np.zeros(pad_shape) # have to look back one
        cell_seq = np.zeros(pad_shape) # have to look back one
        act_o_seq = np.zeros(seq_shape)
        gate_o_seq = np.zeros(seq_shape)
        hidden_seq = np.zeros(pad_shape) # have to look back one
        output_seq = np.zeros(seq_shape)

        for t in xrange(seq_len):
            data = ff_sequence[i,:]
            act_i_seq[t,:] = self.w_x_i * data + self.w_h_i * hidden[t,:] + self.w_c_i * cell[t,:] + self.b_i
            gate_i_seq[t,:] = sig(act_i[t,:])

            act_f_seq[t,:] = self.w_x_f * data + self.w_h_f * hidden[t,:] + self.w_c_f * cell[t,:] + self.b_f
            gate_f_seq[t,:] = sig(act_f_seq[t,:])

            act_c_seq[t+1,:] = self.w_x_c * data + self.w_h_c * hidden_seq[t,:] + self.b_c
            cell_seq[t+1,:] = gate_f_seq[t,:] * cell_seq[t,:] + gate_i_seq[t,:] * tanh(act_c_seq[t+1,:])

            act_o_seq[t,:] = self.w_x_o * data + self.w_h_o * hidden_seq[t,:] + self.w_c_o * cell_seq[t+1,:] + self.b_o
            gate_o_seq[t,:] = sig(act_o_seq[t,:])

            hidden_seq[t+1,:] = gate_o_seq[t,:] * tanh(cell_seq[t+1,:])

            self.output = softmax(self.hidden)
        return ff_sequence, act_i_seq, gate_i_seq, act_f_seq, gate_f_seq, act_c_seq, cell_seq, act_o_seq, gate_o_seq, hidden_seq, output_seq

    def bptt2(self, sequence):
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        seq_shape = sequence.shape
        seq_len = seq_shape[0]
        ff_sequence, act_i_seq, gate_i_seq, act_f_seq, gate_f_seq, act_c_seq, cell_seq, act_o_seq, gate_o_seq, hidden_seq, output_seq = self.ff_seq(sequence)

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
        g_c = np.zeros(self.data_len)
        
        # link = np.zeros(seq_shape)

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

        # these are the terms used to get the partial derivatives from the t=1
        # elem as well
        jump_h = np.zeros(self.data_len)
        jump_c = np.zeros(self.data_len)
        res1 = np.zeros(self.data_len)
        # Now with more 80/81 character limit
        # >tfw mobile coding
        for t in xrange(seq_len, -1, -1):
            data = ff_seq[t,:]
            target = sequence[t,:]
            res0 = hidden_seq[t+1,:] - target
            start = res0 + jump_h

            delta_o = start * dsig(act_o[t,:])
            g_x_o += delta_o * data
            g_h_o += delta_o * hidden_seq[t,:]
            g_c_o += delta_o * cell_seq[t,:]
            g_o += delta_o

            link = tanh(cell[t,:]) * dsig(act_o_seq[t,:] * self.w_c_o
            link += gate_o_seq[t,:] * dtanh(cell[t,:])

            start = (start + jump_c)

            delta_c = start * gate_i_seq[t,:] * dsig(act_c_seq[t,:])
            g_x_c += delta_o * data
            g_h_c += delta_o * hidden_seq[t,:]
            g_c += delta_c

            delta_f = start * cell_seq[t-1,:] * dsig(act_f_seq[t,:])
            g_x_f += delta_f * data
            g_h_f += delta_f * hidden_seq[t,:]
            g_c_f += delta_f * cell_seq[t,:]
            g_f += delta_f

            delta_i = start * tanh(act_c[t,:]) * dsig(act_i[t,:])
            g_x_i += delta_i * data
            g_h_i += delta_i * hidden_seq[t,:]
            g_c_i += delta_i * cell_seq[t,:]
            g_i += delta_i

            jump_h = delta_o * self.w_h_o + delta_c * self.w_h_c 
            jump_h += delta_f * self.w_h_f + delta_i * self.w_h_i
            jump_c = link * gate_f[i,:] + delta_f * self.w_c_f
            jump_c += delta_i * self.w_c_i
            res1 = np.copy(res0)
 
    def bptt(self, sequence):
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        seq_shape = sequence.shape
        seq_len = seq_shape[0]

        # I don't think I need this if I use ff_seq
        # that's because I don't, all taken care of in ff_seq now
        # the sequence to be fed into feedforward, starts out with all zeros
        ff_seq = np.vstack((np.zeros(self.data_len),  sequence))

        # delta & gradients for the output gate
        delta_o = np.zeros(seq_shape)
        g_x_o = np.zeros(seq_shape)
        g_h_o = np.zeros(seq_shape)
        g_c_o = np.zeros(seq_shape)
        g_o = np.zeros(seq_shape)

        # delta & gradients for the cell state, and a fast link to the cell state
        delta_c = np.zeros(seq_shape)
        g_x_c = np.zeros(seq_shape)
        g_h_c = np.zeros(seq_shape)
        g_c = np.zeros(seq_shape)
        
        link = np.zeros(seq_shape)

        # delta & gradients for the forget gate
        delta_f = np.zeros(seq_shape)
        g_x_f = np.zeros(seq_shape)
        g_h_f = np.zeros(seq_shape)
        g_c_f = np.zeros(seq_shape)
        g_f = np.zeros(seq_shape)


        # delta & gradients for the input gate
        delta_i = np.zeros(seq_shape)
        g_x_i = np.zeros(seq_shape)
        g_h_i = np.zeros(seq_shape)
        g_c_i = np.zeros(seq_shape)
        g_i = np.zeros(seq_shape)
        
        for t in xrange(seq_len):
            ff_data = ff_seq[t,:]
            # print ff_data
            # raw_input('hold')
            target = sequence[t,:]

            residue = self.ff(ff_data) - target

            # underflow errors are my favorite thing
            # tbh fam I should have seen this coming
            # ALEX, YOU SAID THIS METHOD AVOIDED VANISHING GRADIENTS
            # ;_;
            try:
                delta_o = residue * tanh(self.cell) * dsig(self.act_o)
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if abs(residue[i]) < 1e-3 or abs(tanh(self.cell[i])) < 1e-3 or abs(dsig(self.act_o[i])) < 1e-3:
                        delta_o[i] = 0
                    else:
                        delta_o[i] = residue[i] * tanh(self.cell[i]) * dsig(self.act_o[i])
            for i in xrange(self.data_len):
                if abs(delta_o[i]) < 1e-8:
                    delta_o[i] = 0
            try:
                g_x_o += delta_o * ff_data
            except FloatingPointError:
                print delta_o
                print ff_data
                raw_input('hold - g_x_o')
            try:
                g_h_o += delta_o * self.hidden_old
            except FloatingPointError:
                print delta_o
                print self.hidden_old
                raw_input('hold - g_h_o')
            try:
                g_c_o += delta_o * self.cell
            except FloatingPointError:
                print delta_o
                print self.cell
                raw_input('hold - g_c_o')
            try:
                g_o += delta_o
            except FloatingPointError:
                print delta_o
                raw_input('hold - g_o')

            try:
                arg1 = tanh(self.cell) * dsig(self.act_o) * self.w_c_o
                for i in xrange(self.data_len):
                    if abs(arg1[i]) < 1e-4:
                        arg1[i] = 0
            except FloatingPointError:
                arg1 = np.zeros(self.data_len)
                for i in xrange(self.data_len):
                    if math.log10(tanh(self.cell[i])) + math.log10(dsig(self.act_o[i])) + math.log10(self.w_c_o[i]) < -4:
                        arg1[i] = 0
                    else:
                        arg1[i] = tanh(self.cell[i]) * dsig(self.act_o[i]) * self.w_c_o[i]
            try:
                arg2 = self.gate_o * dtanh(self.cell)
                for i in xrange(self.data_len):
                    if abs(arg2[i]) < 1e-4:
                        arg2[i] = 0
            except FloatingPointError:
                arg2 = np.zeros(self.data_len)
                for i in xrange(self.data_len):
                    if math.log10(self.gate_o[i]) + math.log10(dtan(self.cell[i])) < 1e-4:
                        arg2[i] = 0
                    else:
                        arg2[i] = self.gate_o[i] * dtanh(self.cell[i])
           
            try:
                link = residue * (arg1 + arg2)
                for i in xrange(self.data_len):
                    if abs(link[i]) < 1e-4:
                        link[i] = 0
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if abs(residue[i]) < 1e-3:
                        link[i] = 0

            try:
                delta_c = link * self.gate_i * dtanh(self.act_c)
                for i in xrange(self.data_len):
                    if abs(delta_c[i]) < 1e-8:
                        delta_c[i] = 0
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if math.log10(link[i]) + math.log10(self.gate_i[i]) + math.log10(self.act_c[i]) < -8:
                        delta_c[i] = 0
                    else:
                        delta_c[i] = link[i] * self.gate_i[i] * dtanh(self.act_c[i])
                
            try:
                g_x_c += delta_c * ff_data
            except FloatingPointError:
                print delta_c
                print ff_data
                raw_input('hold - g_x_c')
            try:
                g_h_c += delta_c * self.hidden_old
            except FloatingPointError:
                print delta_c
                print self.hidden_old
                raw_input('hold - g_h_c')
                g_c += delta_c

            delta_f = link * self.cell_old * dsig(self.act_c)
            for i in xrange(self.data_len):
                if abs(delta_f[i]) < 1e-8:
                    delta_f[i] = 0
            try:
                g_x_f += delta_f * ff_data
            except FloatingPointError:
                print delta_f
                print ff_data
                raw_input('hold - g_x_f')
            try:
                g_h_f += delta_f * self.hidden_old
            except FloatingPointError:
                print delta_f
                print self.hidden_old
                raw_input('hold - g_h_f')
            try:
                g_c_f += delta_f * self.cell_old
            except FloatingPointError:
                print delta_f
                print self.cell_old
                raw_input('hold - g_c_f')
            try:
                g_f += delta_f
            except FloatingPointError:
                print delta_f
                raw_input('hold - g_f')

            delta_i = link * tanh(self.act_c) * dsig(self.act_i)
            for i in xrange(self.data_len):
                if abs(delta_i[i]) < 1e-8:
                    delta_i[i] = 0
            try:
                g_x_i += delta_i * ff_data
            except FloatingPointError:
                print delta_i
                print ff_data
                raw_input('hold - g_x_i')
            try:
                g_h_i += delta_i * self.hidden_old
            except FloatingPointError:
                print delta_i
                print self.hidden_old
                raw_input('hold - g_h_i')
            try:
                g_c_i += delta_i * self.cell_old
            except FloatingPointError:
                print delta_i
                print self.cell_old
                raw_input('hold - g_c_i')
            try:
                g_i += delta_i
            except FloatingPointError:
                print delta_i
                raw_input('hold - g_i')

        return g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o

    def sample(self, sample_len):
        result = np.zeros((sample_len, self.data_len))
        seed = np.zeros(self.data_len)
        for i in xrange(sample_len):
            result[i,:] = oh.hot(np.argmax(self.ff(seed)), self.data_len)
            seed = result[i,:]
        return result

    def train(self, sequence, verbose=True):
        self.adagrad(sequence)
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
        ff_seq = np.vstack((np.zeros(self.data_len), sequence))
        loss = 0
        for t in xrange(len(sequence[0])):
            ff_data = ff_seq[t,:]
            actual = sequence[t,:]
            key = int(np.argmax(actual))
            output = self.ff(ff_data)
            loss -= math.log(output[key])
        if verbose:
            print "current sequence loss: %f" % (loss)
        return loss

    def adagrad(self, sequence):
        g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o = self.bptt2(sequence)

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
        s_c = self.master_step / np.sqrt((step_tol * ones) + self.hb_c)

        s_x_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_x_o)
        s_h_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_h_o)
        s_c_o = self.master_step / np.sqrt((step_tol * ones) + self.hw_c_o)
        s_o = self.master_step / np.sqrt((step_tol * ones) + self.hb_o)

        self.w_x_i -= c_x_i * g_x_i
        self.w_h_i -= s_h_i * g_h_i
        self.w_c_i -= s_c_i * g_c_i
        self.b_i -= s_i * g_i

        self.w_x_f -= s_x_f * g_x_f
        self.w_h_f -= s_h_f * g_h_f
        self.w_c_f -= s_c_f * g_c_f
        self.b_f -= s_f * g_f

        self.w_x_c -= s_x_c * g_x_c
        self.w_h_c -= s_h_c * g_h_c
        self.b_c -= s_c * b_c

        self.w_x_o -= s_x_o * g_x_o
        self.w_h_o -= s_h_o * g_h_o
        self.w_c_o -= s_c_o * g_c_o
        self.b_o -= s_o * g_o

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
        self.b_c += np.square(self.b_c)

        self.hw_x_o += np.square(self.w_x_o)
        self.hw_h_o += np.square(self.w_h_o)
        self.hw_c_o += np.square(self.w_c_o)
        self.hb_o += np.square(self.b_o)
