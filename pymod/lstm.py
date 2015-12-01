#!usr/bin/py

import numpy as np
import onehot as oh
import math

def rand_ND(shape):
    # dude, typing np.random.random_sample is not fun
    rand = np.random.random_sample(shape)
    return rand

def log_sum(*elems):
    return np.sum(np.log10(np.fabs(np.array(elems))))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    try:
        return 1 - np.square(tanh(x))
    except FloatingPointError:
        if type(x) == float or type(x) == int or type(x) == np.float64:
            return 1
        else:
            res = np.zeros(len(x))
            for i in xrange(len(x)):
                arg = tanh(x[i])
                if abs(arg) < 1e-8:
                    arg = 0
                res[i] = 1 - np.square(arg)
            return res
            
# def sig(x):
    # return 1 / (1 + np.exp(-x))
def sig(x):
    np.seterr(over='raise', under='raise')
    if type(x) == float or type(x) == int or type(x) == np.float64:
        if x > 4e2:
            return 1
        elif x < -4e2:
            return 0
        else:
            return 1 / (1 + math.exp(-x))
    else:
        vec_len = len(x)
        try:
            return 1 / (1 + np.exp(-x))
        except FloatingPointError:
            result = np.zeros(len(x))
            for i in xrange(vec_len):
                val = x[i]
                if val > 4e2:
                    result[i] = 1
                elif val < -4e2:
                    pass
                else:
                    result[i] = 1 / (1 + math.exp(-val))
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

        self.master_step = 1e-5

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
        self.w_x_i = np.zeros(data_len)
        self.w_h_i = np.zeros(data_len)
        self.w_c_i = np.zeros(data_len)
        self.b_i = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_i = np.square(self.w_x_i)
        self.hw_h_i = np.square(self.w_h_i)
        self.hw_c_i = np.square(self.w_c_i)
        self.hb_i = np.square(self.b_i)

        # and the last gradients used for momentum
        self.l_x_i = np.zeros(data_len)
        self.l_h_i = np.zeros(data_len)
        self.l_c_i = np.zeros(data_len)
        self.l_i = np.zeros(data_len)

        # defines the forget gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_f = np.zeros(data_len)
        self.act_f = np.zeros(data_len)
        self.w_x_f = np.zeros(data_len)
        self.w_h_f = np.zeros(data_len)
        self.w_c_f = np.zeros(data_len)
        self.b_f = np.ones(data_len) # supposedly works better with ones

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_f = np.square(self.w_x_f)
        self.hw_h_f = np.square(self.w_h_f)
        self.hw_c_f = np.square(self.w_c_f)
        self.hb_f = np.square(self.b_f)

        # and the last gradients used for momentum
        self.l_x_f = np.zeros(data_len)
        self.l_h_f = np.zeros(data_len)
        self.l_c_f = np.zeros(data_len)
        self.l_f = np.zeros(data_len)

        # defines the cell state, the argument of its activation function, and the biases and weights used to calculate it
        self.cell = np.zeros(data_len)
        self.act_c = np.zeros(data_len)
        self.w_x_c = np.zeros(data_len)
        self.w_h_c = np.zeros(data_len)
        self.b_c = np.zeros(data_len)

        # defines the former cell state
        # self.cell_past = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_c = np.square(self.w_x_c)
        self.hw_h_c = np.square(self.w_h_c)
        self.hb_c = np.square(self.b_c)

        # and the last gradients used for momentum
        self.l_x_c = np.zeros(data_len)
        self.l_h_c = np.zeros(data_len)
        self.l_c = np.zeros(data_len)

        # defines the output gate, the argument of its activation function, and the biases and weights used to calculate it
        self.gate_o = np.zeros(data_len)
        self.act_o = np.zeros(data_len)
        self.w_x_o = np.zeros(data_len)
        self.w_h_o = np.zeros(data_len)
        self.w_c_o = np.zeros(data_len)
        self.b_o = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_x_o = np.square(self.w_x_o)
        self.hw_h_o = np.square(self.w_h_o)
        self.hw_c_o = np.square(self.w_c_o)
        self.hb_o = np.square(self.b_o)

        # and the last gradients used for momentum
        self.l_x_o = np.zeros(data_len)
        self.l_h_o = np.zeros(data_len)
        self.l_c_o = np.zeros(data_len)
        self.l_o = np.zeros(data_len)

        # the hidden output of the lstm, and the previous hidden output
        # self.hidden_old = np.zeros(data_len)
        self.hidden = np.zeros(data_len)
        self.output = np.zeros(data_len)

    # feedforward data through the network
    # only used for the sequence loss
    def ff(self, data, verbose=True):
        if len(data) != self.data_len:
            raise ValueError("Unexpected data length during feedforward")

        try:
            self.act_i = self.w_x_i * data + self.w_h_i * self.hidden + self.w_c_i * self.cell
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.w_x_i[i], data[i]) < 1e-40 :
                    self.act_i[i] = 0
                else:
                    self.act_i[i] = f.w_x_i[i] * data[i]
                if log_sum(self.w_h_i[i], self.hidden[i]) < 1e-40 :
                    self.act_i[i] = 0
                else:
                    self.act_i[i] += self.w_h_i[i] * self.hidden[i]
                if log_sum(self.w_c_i[i], self.cell) < 1e-40 :
                    self.act_i[i] = 0
                else:
                    self.act_i[i] += self.w_c_i[i] * self.cell
        self.act_i += self.b_i
        self.gate_i = sig(self.act_i)

        try:
            self.act_f = self.w_x_f * data + self.w_h_f * self.hidden + self.w_c_f * self.cell
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.w_x_f[i], data[i]) < 1e-40 :
                    self.act_f[i] = 0
                else:
                    self.act_f[i] = self.w_x_f[i] * data[i]
                if log_sum(self.w_h_f[i], self.hidden[i]) < 1e-40 :
                    continue
                else:
                    self.act_f[i] += self.w_h_f[i] * self.hidden[i]
                if log_sum(self.w_c_f[i], self.cell[i]) < 1e-40 :
                    continue
                else:
                    self.act_f[i] += self.w_c_f[i] * self.cell[i]
        self.act_f += self.b_f
        self.gate_f = sig(self.act_f)

        try:
            self.act_c = self.w_x_c * data + self.w_h_c * self.hidden
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.w_x_c[i], data[i]) < 1e-40 :
                    self.act_c[i] = 0
                else:
                    self.act_c[i] = self.w_x_c[i] * data[i]
                if log_sum(self.w_h_c[i], self.hidden[i]) < 1e-40 :
                    continue
                else:
                    self.act_c[i] += self.w_h_c[i] * self.hidden[i]
        self.act_c += self.b_c

        try:
            self.cell = self.gate_f * self.cell + self.gate_i * tanh(self.act_c)
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.gate_f[i], self.cell[i]) < 1e-40 :
                    self.cell[i] = 0
                else:
                    self.cell[i] = self.gate_f[i] * self.cell[i]
                if log_sum(self.gate_i[i], tanh(self.act_c[i])) < 1e-40 :
                    continue
                else:
                    self.cell[i] += self.gate_i[i] * tanh(self.act_c[i])

        try:
            self.act_o = self.w_x_o * data + self.w_h_o * self.hidden + self.w_c_o * self.cell
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.w_x_o[i], data[i]) < 1e-40 :
                    self.act_o[i] = 0
                else:
                    self.act_o[i] = self.w_x_o[i] * data[i]
                if log_sum(self.w_h_o[i], self.hidden[i]) < 1e-40 :
                    continue
                else:
                    self.act_o[i] += self.w_h_o[i] * self.hidden[i]
                if log_sum(self.w_c_o[i], self.cell[i]) < 1e-40 :
                    continue
                else:
                    self.act_o[i] += self.w_c_o[i] * self.cell[i]
        self.act_o += self.b_o
        self.gate_o = sig(self.act_o)

        try:
            self.hidden = self.gate_o * tanh(self.cell)
        except FloatingPointError:
            for i in xrange(self.data_len):
                if log_sum(self.gate_o[i], tanh(self.cell[i])) < 1e-40:
                    self.hidden[i] = 0
                else:
                    self.hidden[i] = self.gate_o[i] * tanh(self.cell[i])


        self.output = softmax(self.hidden)
        if verbose:
            return self.output

    def ff_seq(self, sequence):
        if sequence.shape[1] != self.data_len:
            raise ValueError("Unexpected data length during sequence feedforward")
        ff_sequence = np.vstack((np.zeros(self.data_len), sequence))
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
            data = ff_sequence[t,:]
            try:
                act_i_seq[t,:] = self.w_x_i * data + self.w_h_i * hidden_seq[t,:] + self.w_c_i * cell_seq[t,:] + self.b_i
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(self.w_x_i[i], data[i]) < 1e-40:
                        continue
                    else:
                        act_i_seq[t,i] = self.w_x_i[i] * data[i]
                    if log_sum(self.w_h_i[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_i_seq[t,i] += self.w_h_i[i] * hidden_seq[t,i]
                    if log_sum(self.w_c_i[i], cell_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_i_seq[t,i] += self.w_c_i[i] * cell_seq[t,i]
                    act_i_seq += self.b_i[i]
            gate_i_seq[t,:] = sig(act_i_seq[t,:])

            try:
                act_f_seq[t,:] = self.w_x_f * data + self.w_h_f * hidden_seq[t,:] + self.w_c_f * cell_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(self.w_x_f[i], data[i]) < 1e-40:
                        continue
                    else:
                        act_f_seq[t,i] = self.w_x_f[i] * data[i]
                    if log_sum(self.w_h_f[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_f_seq[t,i] += self.w_h_f[i] * hidden_seq[t,i]
                    if log_sum(self.w_c_f[i], cell_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_f_seq[t,i] += self.w_c_f[i] * cell_seq[t,i]
            act_f_seq[t,:] += self.b_f
            gate_f_seq[t,:] = sig(act_f_seq[t,:])

            try:
                act_c_seq[t+1,:] = self.w_x_c * data + self.w_h_c * hidden_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(self.w_x_c[i], data[i]) < 1e-40:
                        continue
                    else:
                        act_c_seq[t+1,i] = self.w_x_c[i] * data[i]
                    if log_sum(self.w_h_c[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_c_seq[t+1,i] += self.w_h_c[i] * hidden_seq[t,i]
            act_c_seq[t+1,:] += self.b_c
            try:
                cell_seq[t+1,:] =  gate_i_seq[t,:] * tanh(act_c_seq[t+1,:]) + gate_f_seq[t,:] * cell_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(gate_i_seq[t,i], tanh(act_c_seq[t+1,i])) < 1e-40:
                        continue
                    else:
                        cell_seq[t+1,i] = gate_i_seq[t,i] * tanh(act_c_seq[t+1,i])
                    if log_sum(gate_f_seq[t,i], cell_seq[t,i]) < 1e-40:
                        continue
                    else:
                        cell_seq[t+1,i] += gate_f_seq[t,i] * cell_seq[t,i]

            try:
                act_o_seq[t,:] = self.w_x_o * data + self.w_h_o * hidden_seq[t,:] + self.w_c_o * cell_seq[t+1,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(self.w_x_o[i], data[i]) < 1e-40:
                        continue
                    else:
                        act_o_seq[t,i] = self.w_x_o[i] * data[i]
                    if log_sum(self.w_h_o[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        act_o_seq[t,i] += self.w_h_o[i] * hidden_seq[t,i]
                    if log_sum(self.w_c_o[i], cell_seq[t+1,i]) < 1e-40:
                        continue
                    else:
                        act_o_seq[t,i] += self.w_c_o[i] * cell_seq[t+1,i]
            act_o_seq[t,:] += self.b_o
            gate_o_seq[t,:] = sig(act_o_seq[t,:])

            try:
                hidden_seq[t+1,:] = gate_o_seq[t,:] * tanh(cell_seq[t+1,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(gate_o_seq[t,i], tanh(cell_seq[t+1,i])) < 1e-40:
                        continue
                    else:
                        hidden_seq[t+1,i] = gate_o_seq[t,i] * tanh(cell_seq[t+1,i])

            self.output = softmax(self.hidden)
        return ff_sequence, act_i_seq, gate_i_seq, act_f_seq, gate_f_seq, act_c_seq, cell_seq, act_o_seq, gate_o_seq, hidden_seq, output_seq

    def bptt(self, sequence, dense=False):
        if len(sequence[0]) != self.data_len:
            raise ValueError("Unexpected data length during bptt")
        seq_shape = sequence.shape
        seq_len = seq_shape[0]
        ff_sequence, act_i_seq, gate_i_seq, act_f_seq, gate_f_seq, act_c_seq, cell_seq, act_o_seq, gate_o_seq, hidden_seq, output_seq = self.ff_seq(sequence)

        # gradients for the output gate
        delta_o = np.zeros(self.data_len)
        g_x_o = np.zeros(self.data_len)
        g_h_o = np.zeros(self.data_len)
        g_c_o = np.zeros(self.data_len)
        g_o = np.zeros(self.data_len)

        # gradients for the cell state, and a fast link to the cell state
        delta_c = np.zeros(self.data_len)
        g_x_c = np.zeros(self.data_len)
        g_h_c = np.zeros(self.data_len)
        g_c = np.zeros(self.data_len)

        link = np.zeros(self.data_len)
        
        # gradients for the forget gate
        delta_f = np.zeros(self.data_len)
        g_x_f = np.zeros(self.data_len)
        g_h_f = np.zeros(self.data_len)
        g_c_f = np.zeros(self.data_len)
        g_f = np.zeros(self.data_len)

        # gradients for the input gate
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
        for t in xrange(seq_len - 1, -1, -1):
            data = ff_sequence[t,:]
            target = sequence[t,:]
            res = hidden_seq[t+1,:] - target
            start = res + jump_h

            # Output Gate
            try:
                delta_o = start * tanh(cell_seq[t+1,:]) * dsig(act_o_seq[t,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(start[i], dsig(act_o_seq[t,i]), tanh(cell_seq[t+1,i])) < 1e-40:
                        delta_o[i] = 0
                    else:
                        delta_o[i] = start[i] * tanh(cell_seq[t+1,i]) * dsig(act_o_seq[t,i])
            g_x_o += delta_o * data
            try:
                g_h_o += delta_o * hidden_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_o[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        g_h_o[i] += delta_o[i] * hidden_seq[t,i]
            try:
                g_c_o += delta_o * cell_seq[t+1,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_o[i], cell_seq[t+1,i]) < 1e-40:
                        continue
                    else:
                        g_c_o[i] += delta_o[i] * cell_seq[t+1,i]
            g_o += delta_o

            # Link Update
            try:
                link = tanh(cell_seq[t+1,:]) * dsig(act_o_seq[t,:]) * self.w_c_o + gate_o_seq[t,:] * dtanh(cell_seq[t+1,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(tanh(cell_seq[t+1,i]), dsig(act_o_seq[t,i]), self.w_c_o[i]) < 1e-40:
                        link[i] = 0
                    else:
                        link[i] = tanh(cell_seq[t+1,i]) * dsig(act_o_seq[t,i]) * self.w_c_o[i]
                    if log_sum(gate_o_seq[t,i], dtanh(cell_seq[t+1,i])) < 1e-40:
                        continue
                    else:
                        link[i] += gate_o_seq[t,i] * dtanh(cell_seq[t+1,i])

            # Start Update
            try:
                start = start * link
            except:
                for i in xrange(self.data_len):
                    if log_sum(start[i], link[i]) < 1e-40:
                        start[i] = 0
                    else:
                        start[i] = start[i] * link[i]
            start += jump_c

            # Cell state
            try:
                delta_c = start * gate_i_seq[t,:] * dsig(act_c_seq[t+1,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(start[i], gate_i_seq[t,i], dsig(act_c_seq[t+1,i])) < 1e-40:
                        delta_c[i] = 0
                    else:
                        delta_c[i] = start[i] * gate_i_seq[t,i] * dsig(act_c_seq[t+1,i])
            g_x_c += delta_c * data
            try:
                g_h_c += delta_c * hidden_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_c[i], hidden_seq[t,i]):
                        continue
                    else:
                        g_h_c[i] += delta_c[i] * hidden_seq[t,i]
            g_c += delta_c

            # Forget Gate
            try:
                delta_f = start * cell_seq[t,:] * dsig(act_f_seq[t,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(start[i], cell_seq[t,i], dsig(act_f_seq[t,i])) < 1e-40:
                        delta_f[i] = 0
                    else:
                        delta_f[i] = start[i] * cell_seq[t,i] * dsig(act_f_seq[t,i])
            g_x_f += delta_f * data
            try:
                g_h_f += delta_f * hidden_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_f[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        g_h_f[i] += delta_f[i] * hidden_seq[t,i]
            try:
                g_c_f += delta_f * cell_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_f[i], cell_seq[t,i]) < 1e-40:
                        continue
                    else:
                        g_c_f[i] = delta_f[i] * cell_seq[t,i]
            g_f += delta_f

            # Input Gate
            try:
                delta_i = start * tanh(act_c_seq[t,:]) * dsig(act_i_seq[t,:])
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(start[i], tanh(act_c_seq[t,i]), dsig(act_i_seq[t,i])) < 1e-40:
                        delta_i[i] = 0
                    else:
                        delta_i[i] = start[i] * tanh(act_c_seq[t,i]) * dsig(act_i_seq[t,i])
            g_x_i += delta_i * data
            try:
                g_h_i += delta_i * hidden_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_i[i], hidden_seq[t,i]) < 1e-40:
                        continue
                    else:
                        g_h_i[i] = delta_i[i] * hidden_seq[t,i]
            try:
                g_c_i += delta_i * cell_seq[t,:]
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(delta_i[i], cell_seq[t,i]) < 1e-40:
                        continue
                    else:
                        g_c_i[i] = delta_i[i] * cell_seq[t,i]
            g_i += delta_i

            # Former Hidden Jump
            try:
                jump_h = delta_o * self.w_h_o + delta_c * self.w_h_c + delta_f * self.w_h_f + delta_i * self.w_h_i
            except FloatingPointError:
                for i in xrange(self.data_len):                
                    if log_sum(delta_o[i], self.w_h_o[i]) < 1e-40:
                        jump_h[i] = 0
                    else:
                        jump_h[i] = delta_o[i] * self.w_h_o[i]
                    if log_sum(delta_c[i], self.w_h_c[i]) < 1e-40:
                        continue
                    else:
                        jump_h[i] = delta_c[i] * self.w_h_c[i]
                    if log_sum(delta_f[i], self.w_h_f[i]) < 1e-40:
                        continue
                    else:
                        jump_h[i] += delta_f[i] * self.w_h_f[i]
                    if log_sum(delta_i[i], self.w_h_i[i]) < 1e-40:
                        continue
                    else:
                        jump_h[i] += delta_i[i] * self.w_h_i[i]

            # Former Cell Jump
            try:
                jump_c = start * gate_f_seq[t,:] + delta_f * self.w_c_f + delta_i * self.w_c_i
            except FloatingPointError:
                for i in xrange(self.data_len):
                    if log_sum(link[i], gate_f_seq[t,i]) < 1e-40:
                        jump_c[i] = 0
                    else:
                        jump_c[i] = link[i] * gate_f_seq[t,i]
                    if log_sum(delta_f[i], self.w_c_f[i]) < 1e-40:
                        continue
                    else:
                        jump_c[i] += delta_f[i] * self.w_c_f[i]
                    if log_sum(delta_i[i], self.w_c_i[i]) < 1e-40:
                        continue
                    else:
                        jump_c[i] += delta_i[i] * self.w_c_i[i]

        # print g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o
        # raw_input('hold')
        if dense:
            return [g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o]
        else:
            return g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o
 
    def sample(self, sample_len):
        self.reset()
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
        self.reset()
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

    def adagrad(self, sequence, forget=0.8):
        g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o = self.bptt(sequence)

        ones = np.ones(self.data_len)

        step_tol = 1e-1

        #calculate all the step sizes
        s_x_i = self.master_step / (step_tol + np.sqrt(self.hw_x_i))
        s_h_i = self.master_step / (step_tol + np.sqrt(self.hw_h_i))
        s_c_i = self.master_step / (step_tol + np.sqrt(self.hw_c_i))
        s_i = self.master_step / (step_tol + np.sqrt(self.hb_i))

        s_x_f = self.master_step / (step_tol + np.sqrt(self.hw_x_f))
        s_h_f = self.master_step / (step_tol + np.sqrt(self.hw_h_f))
        s_c_f = self.master_step / (step_tol + np.sqrt(self.hw_c_f))
        s_f = self.master_step / (step_tol + np.sqrt(self.hb_f))

        s_x_c = self.master_step / (step_tol + np.sqrt(self.hw_x_c))
        s_h_c = self.master_step / (step_tol + np.sqrt(self.hw_h_c))
        s_c = self.master_step / (step_tol + np.sqrt(self.hb_c))

        s_x_o = self.master_step / (step_tol + np.sqrt(self.hw_x_o))
        s_h_o = self.master_step / (step_tol + np.sqrt(self.hw_h_o))
        s_c_o = self.master_step / (step_tol + np.sqrt(self.hw_c_o))
        s_o = self.master_step / (step_tol + np.sqrt(self.hb_o))

        self.w_x_i -= s_x_i * (forget * self.l_x_i + g_x_i)
        self.w_h_i -= s_h_i * (forget * self.l_h_i + g_h_i)
        self.w_c_i -= s_c_i * (forget * self.l_c_i + g_c_i)
        self.b_i -= s_i * (forget * self.l_i + g_i)

        self.w_x_f -= s_x_f * (forget * self.l_x_f + g_x_f)
        self.w_h_f -= s_h_f * (forget * self.l_h_f + g_h_f)
        self.w_c_f -= s_c_f * (forget * self.l_c_f + g_c_f)
        self.b_f -= s_f * (forget * self.l_f + g_f)

        self.w_x_c -= s_x_c * (forget * self.l_x_c + g_x_c)
        self.w_h_c -= s_h_c * (forget * self.l_h_c + g_h_c)
        self.b_c -= s_c * (forget * self.l_c + g_c)

        self.w_x_o -= s_x_o * (forget * self.l_x_o + g_x_o)
        self.w_h_o -= s_h_o * (forget * self.l_h_o + g_h_o)
        self.w_c_o -= s_c_o * (forget * self.l_c_o + g_c_o)
        self.b_o -= s_o * (forget * self.l_o + g_o)

        self.last_grads(g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o)

        self.archive_sq_weights()
        self.reset()

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
        self.hb_c += np.square(self.b_c)

        self.hw_x_o += np.square(self.w_x_o)
        self.hw_h_o += np.square(self.w_h_o)
        self.hw_c_o += np.square(self.w_c_o)
        self.hb_o += np.square(self.b_o)

    def last_grads(self, g_x_i, g_h_i, g_c_i, g_i, g_x_f, g_h_f, g_c_f, g_f, g_x_c, g_h_c, g_c, g_x_o, g_h_o, g_c_o, g_o):
        self.l_x_i = g_x_i
        self.l_h_i = g_h_i
        self.l_c_i = g_c_i
        self.l_i = g_i

        self.l_x_f = g_x_f
        self.l_h_f = g_h_f
        self.l_c_f = g_c_f
        self.l_f = g_f

        self.l_x_c = g_x_c
        self.l_h_c = g_h_c
        self.l_c = g_c

        self.l_x_o = g_x_o
        self.l_h_o = g_h_o
        self.l_c_o = g_c_o
        self.l_o = g_o

    def reset(self):
        self.gate_i = np.zeros(self.data_len)
        self.act_i = np.zeros(self.data_len)
        self.gate_f = np.zeros(self.data_len)
        self.act_f = np.zeros(self.data_len)
        self.cell = np.zeros(self.data_len)
        self.act_c = np.zeros(self.data_len)
        self.gate_o = np.zeros(self.data_len)
        self.act_o = np.zeros(self.data_len)
        self.hidden = np.zeros(self.data_len)
        self.output = np.zeros(self.data_len)

    def memetrain(self, gens, pop_size, num_saved, num_new, x_rate, m_rate):
        ayy = 0
        for g in xrange(gens):
            lmao = 0
