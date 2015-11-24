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

# an deep RNN inspired by Alex Graves' own work with RNNs of LSTMs
class DeepLSTM:

    def __init__(self, data_len):
        self.data_len = int(data_len)

        self.data = rand_ND(data_len) # stores the most recent sequence data given to the LSTM

        self.output = rand_ND(data_len) # stores the most recent 

        # Why aren't I defining an LSTM object?  I think bptt might be wonky between objects
        # gonna hardcode first

        ###############
        #    LSTM0    #
        ###############

        # fed in the input data--a single observation in a sequence
        # its output gets fed into LSTM1

        # defines the input gate, and the biases and weights used to calculate it for lstm0
        self.gate_i0 = rand_ND(data_len)
        self.w_d_i0 = rand_ND(data_len)
        self.w_h_i0 = rand_ND(data_len)
        self.w_c_i0 = rand_ND(data_len)
        self.b_i0 = rand_ND(data_len)

        # and their gradients
        self.gw_d_i0 = np.zeros(data_len)
        self.gw_h_i0 = np.zeros(data_len)
        self.gw_c_i0 = np.zeros(data_len)
        self.gb_i0 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_i0 = np.zeros(data_len)
        self.hw_h_i0 = np.zeros(data_len)
        self.hw_c_i0 = np.zeros(data_len)
        self.hb_i0 = np.zeros(data_len)

        # defines the forget gate, and the biases and weights used to calculate it for lstm0
        self.gate_f0 = rand_ND(data_len)
        self.w_d_f0 = rand_ND(data_len)
        self.w_h_f0 = rand_ND(data_len)
        self.w_c_f0 = rand_ND(data_len)
        self.b_f0 = rand_ND(data_len)

        # and their gradients
        self.gw_d_f0 = np.zeros(data_len)
        self.gw_h_f0 = np.zeros(data_len)
        self.gw_c_f0 = np.zeros(data_len)
        self.gb_f0 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_f0 = np.zeros(data_len)
        self.hw_h_f0 = np.zeros(data_len)
        self.hw_c_f0 = np.zeros(data_len)
        self.hb_f0 = np.zeros(data_len)

        # defines the cell state, and the biases and weights used to calculate it for lstm0
        self.cell0 = rand_ND(data_len)
        self.w_d_c0 = rand_ND(data_len)
        self.w_h_c0 = rand_ND(data_len)
        self.b_c0 = rand_ND(data_len)

        # and their gradients 
        self.gw_d_c0 = np.zeros(data_len)
        self.gw_h_c0 = np.zeros(data_len)
        self.gb_c0 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_c0 = np.zeros(data_len)
        self.hw_h_c0 = np.zeros(data_len)
        self.hb_c0 = np.zeros(data_len)

        # defines the output gate, and the biases and weights used to calculate it for lstm0
        self.gate_o0 = rand_ND(data_len)
        self.w_d_o0 = rand_ND(data_len)
        self.w_h_o0 = rand_ND(data_len)
        self.w_c_o0 = rand_ND(data_len)
        self.b_o0 = rand_ND(data_len)

        # and their gradients
        self.gw_d_o0 = np.zeros(data_len)
        self.gw_h_o0 = np.zeros(data_len)
        self.gw_c_o0 = np.zeros(data_len)
        self.gb_o0 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_o0 = np.zeros(data_len)
        self.hw_h_o0 = np.zeros(data_len)
        self.hw_c_o0 = np.zeros(data_len)
        self.hb_o0 = np.zeros(data_len)

        # the hidden output of lstm0
        self.hidden0 = rand_ND(data_len)

        ##############
        #   LSTM1    #
        ##############

        # fed in output from LSTM0
        # output gets softmaxed
        # then finds max val in softmaxed output
        # then gets returned as the one-hot of the argmax of the softmax
        # keep the og output though for adagrad

        # defines the input gate, and the biases and weights used to calculate it for lstm1
        self.gate_i1 = rand_ND(data_len)
        self.w_d_i1 = rand_ND(data_len)
        self.w_h_i1 = rand_ND(data_len)
        self.w_c_i1 = rand_ND(data_len)
        self.b_i1 = rand_ND(data_len)

        # and their gradients
        self.gw_d_i1 = np.zeros(data_len)
        self.gw_h_i1 = np.zeros(data_len)
        self.gw_c_i1 = np.zeros(data_len)
        self.gb_i1 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        # h is for history!
        self.hw_d_i1 = np.zeros(data_len)
        self.hw_h_i1 = np.zeros(data_len)
        self.hw_c_i1 = np.zeros(data_len)
        self.hb_i1 = np.zeros(data_len)

        # defines the forget gate, and the biases and weights used to calculate it for lstm1
        self.gate_f1 = rand_ND(data_len)
        self.w_d_f1 = rand_ND(data_len)
        self.w_h_f1 = rand_ND(data_len)
        self.w_c_f1 = rand_ND(data_len)
        self.b_f1 = rand_ND(data_len)

        # and their gradients
        self.gw_d_f1 = np.zeros(data_len)
        self.gw_h_f1 = np.zeros(data_len)
        self.gw_c_f1 = np.zeros(data_len)
        self.gb_f1 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_f1 = np.zeros(data_len)
        self.hw_h_f1 = np.zeros(data_len)
        self.hw_c_f1 = np.zeros(data_len)
        self.hb_f1 = np.zeros(data_len)

        # defines the cell state, and the biases and weights used to calculate it for lstm1
        self.cell1 = rand_ND(data_len)
        self.w_d_c1 = rand_ND(data_len)
        self.w_h_c1 = rand_ND(data_len)
        self.b_c1 = rand_ND(data_len)

        # and their gradients 
        self.gw_d_c1 = np.zeros(data_len)
        self.gw_h_c1 = np.zeros(data_len)
        self.gb_c1 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_c1 = np.zeros(data_len)
        self.hw_h_c1 = np.zeros(data_len)
        self.hb_c1 = np.zeros(data_len)

        # defines the output gate, and the biases and weights used to calculate it for lstm1
        self.gate_o1 = rand_ND(data_len)
        self.w_d_o1 = rand_ND(data_len)
        self.w_h_o1 = rand_ND(data_len)
        self.w_c_o1 = rand_ND(data_len)
        self.b_o1 = rand_ND(data_len)

        # and their gradients
        self.gw_d_o1 = np.zeros(data_len)
        self.gw_h_o1 = np.zeros(data_len)
        self.gw_c_o1 = np.zeros(data_len)
        self.gb_o1 = np.zeros(data_len)

        # and the sum of the elem-wise squares of all past gradients for adagrad
        self.hw_d_o1 = np.zeros(data_len)
        self.hw_h_o1 = np.zeros(data_len)
        self.hw_c_o1 = np.zeros(data_len)
        self.hb_o1 = np.zeros(data_len)

        # the hidden output of lstm1
        self.hidden1 = rand_ND(data_len)

        # stores the previous loss value, in order to calculate the NEXT loss value :(
        self.loss = 0

        #yeah, im defining an LSTM object if I ever improve this project once I'm done

    # feedforward data through the network
    def ff(self, data, verbose=True):
        self.gate_i0 = sig(self.w_d_i0 * data + self.w_h_i0 * self.hidden0 + self.w_c_i0 * self.cell0 + self.b_i0)
        self.gate_f0 = sig(self.w_d_f0 * data + self.w_h_f0 * self.hidden0 + self.w_c_f0 * self.cell0 + self.b_f0)
        self.cell0 = self.gate_f0 * self.cell0 + self.gate_i0 * tanh(self.w_d_c0 * data0 + self.w_h_c0 * self.hidden0 + celf.b_c0)
        self.gate_o0 = sig(self.w_d_o0 * data + self.w_h_o0 * self.hidden0 + self.w_c_o0 * self.cell0 + self.b_o0)
        self.hidden0 = self.gate_o0 * tanh(self.cell0)

        self.gate_i1 = sig(self.w_d_i1 * data + self.w_h_i1 * self.hidden1 + self.w_c_i1 * self.cell1 + self.b_i1)
        self.gate_f1 = sig(self.w_d_f1 * data + self.w_h_f1 * self.hidden1 + self.w_c_f1 * self.cell1 + self.b_f1)
        self.cell1 = self.gate_f1 * self.cell1 + self.gate_i1 * tanh(self.w_d_c1 * data1 + self.w_h_c1 * self.hidden1 + celf.b_c1)
        self.gate_o1 = sig(self.w_d_o1 * data + self.w_h_o1 * self.hidden1 + self.w_c_o1 * self.cell1 + self.b_o1)
        self.hidden1 = self.gate_o1 * tanh(self.cell1)
       
        # at the moment just returns what it thinks is the most likely output
        # I could probably fix this pretty easily
        # I mean, I cache self.hidden1
        # nbd atm tbh fam
        self.output = oh.hot(np.argmax(softmax(self.hidden1)))
        if verbose:
            return self.output

    def bptt(self, target):
        return 'ayylmao'

    def sample(self, sample_len):
        result = np.zeros((self.data_len, sample_len))
        start = np.zeros(self.data_len)
        for i in xrange(sample_len)
            if i == 0:
                result[i,:] = self.ff(start)
            else:
                result[i,:] = self.ff(result[i-1,:])
        return result

    def loss(self, target, verbose=True):

        if verbose:
            return self.loss

    def adagrad(self):
        grad0, grad1, grad2 = self.bptt()

        ones0 = np.ones(self.weight0.shape)
        ones1 = np.ones(self.weight1.shape)
        ones2 = np.ones(self.weight2.shape)

        step_tol = 1e-6

        step0 = self.step / np.sqrt((step_tol * ones0) + self.history0)
        step1 = self.step / np.sqrt((step_tol * ones1) + self.history1)
        step2 = self.step / np.sqrt((step_tol * ones2) + self.history2)

        self.history0 += np.sq(grad0)
        self.history1 += np.sq(grad1)
        self.history2 += np.sq(grad2)

        return step0*grad0, step1*grad1, step2*grad2

    def update_weights(self, w0, w1, w2):
        self.weight0 -= w0
        self.weight1 -= w1
        self.weight2 -= w2
