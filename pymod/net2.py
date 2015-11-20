#!usr/bin/python

#!/usr/bin/py

import numpy as np
import math

def id(x):
    return x

def did(x):
    try:
        return np.ones(len(x))
    except:
        return 1

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - pow(np.tanh(x), 2)

def sig(x):
    try:
        return np.ones(len(x)) / (1 + np.exp(x))
    except:
        return 1 / (1 + np.exp(x))
    
def dsig(x):
    return sig(x) * (1 - sig(x))

class FFNN():
    """ 2 layer neural network that can use multiple (3) methods to train """

    def __init__(self, indata_len, output_len, layer0_len, layer1_len, f_out, df_out)::
        self.indata_len = int(indata_len)
        self.layer0_len = int(layer0_len)
        self.layer1_len = int(layer1_len)
        self.output_len = int(output_len)

        self.indata = np.ones(indata_len + 1)
        self.output = np.zeros(output_len)

        self.activation0 = np.zeros(layer0_len + 1)
        self.activation1 = np.zeros(layer1_len + 1)
        self.activation2 = np.zeros(output_len)

        self.hmap0 = np.zeros(layer0_len + 1)
        self.hmap1 = np.zeros(layer1_len + 1)

        self.shape_w0 = (indata_len + 1, layer0_len)
        self.shape_w1 = (layer0_len + 1, layer1_len)
        self.shape_w2 = (layer1_len + 1, output_len)

        self.weight0 = 2 * np.random.random_sample(self.shape_w0) - 1
        self.weight1 = 2 * np.random.random_sample(self.shape_w1) - 1
        self.weight2 = 2 * np.random.random_sample(self.shape_w2) - 1

        self.g_weight0 = np.zeros(self.shape_w0)
        self.g_weight1 = np.zeros(self.shape_w1)
        self.g_weight2 = np.zeros(self.shape_w2)

        self.f_out = f_out
        self.df_out = df_out

    def feedforward(self, indata):
        """runs the net with given indata and saves hidden activations"""

        if len(indata) != self.indata_len:
            raise ValueError("Input dimensions not match expected dimensions")
        
        self.indata = np.append(indata, 1)
        for j in range(self.layer0_len):
            self.activation0[j] = np.vdot(self.weight0[:,j], self.indata)
        self.hmap0 = np.tanh(self.activation0)
        
        for k in range(self.layer1_len):
            self.activation1[k] = np.vdot(self.weight1[:,k], self.hmap0)
        self.hmap1 = np.tanh(self.activation1)
        
        for l in range(self.output_len):
            self.activation2[l] = np.vdot(self.weight2[:,l], self.hmap1)
        self.output = self.f_out(self.activation2)
        return self.output
 
    def backprop(self, target):
        k_product = np.zeros(self.layer1_len)
        j_product = np.zeros(self.layer0_len)

        delta2 = 2 * np.absolute(self.output - target) * self.df_out(self.activation2) / self.output_len
        for l in xrange(self.output_len):
            for k in xrange(self.layer1_len + 1):
                self.g_weight2[k,l] = delta2[l] * self.hmap1[k]

        delta1 = np.zeros(self.layer1_len)
        for k in xrange(self.layer1_len):
            for  l in xrange(self.output_len):
                delta1[k] += delta2[l] * self.hmap1[k]
            k_product[k] = delta1[l] * dtanh(self.activation1[k])
            for j in xrange(self.layer0_len + 1):
                self.g_weight1[j,k] = k_product[k] * self.hmap0[j]

        delta0 = np.zeros(self.layer0_len)
        for j in range(self.layer0_len):
            for k in range(self.layer1_len):
                delta0[j] += k_product[k] * self.weight1[j,k]
            j_product[j] = delta0[j] * dtanh(self.activation0[j])
            for i in range(self.indata_len + 1):
                self.g_weight0[i,j] = j_product[j] * self.indata[i]

    def result(self):
        return self.output

    # Snake!?
    def err_sumsq(self, data, target):
        self.feedforward(data)
        return sum_sq_err(self.output, target)

    def err_meansq(self, data, target):
        self.feedforward(data)
        return mean_sq_err(self.output, target)

    def curent_err_sumsq(self, target):
        return sum_sq_err(self.output, target)

    def current_err_meansq(self, target):
        return mean_sq_err(self.output, target)

    def train_meme(self, indata, target, trust, verbose=False):
        start_w0 = self.weight0
        start_w1 = self.weight1
        start_w2 = self.weight2

        best_w0 = self.weight0
        best_w1 = self.weight1
        best_w2 = self.weight2

        min_err = self.current_err_meansq(target)
        if min_err < trust:
            if verbose:
                print "update mean square error: %f" % (min_err)
            return

        mags = np.linspace(-6, -2, 5)
        mags = np.power(10, mags)
        leads = np.linspace(1, 5, 2)

        for l0 in leads:
            for l1 in leads:
                for l2 in leads:
                    for m0 in mags:
                        for m1 in mags:
                            for m2 in mags:
                                err_gd = self.gd_err(indata, target, l0*m0, l1*m1, l2*m2)
                                if err_gd < min_err:
                                    min_err = err_gd
                                    best_w0, best_w1, best_w2 = self.get_weights()
                                self.set_weights(start_w0, start_w1, start_w2)
                                for i in xrange(4):
                                    err_rand = self.rand_err(indata, target, l0*m0, l1*m1, l2*m2)
                                    if err_rand < min_err:
                                        min_err = err_rand
                                        best_w0, best_w1, best_w2 = self.get_weights()
                                    self.set_weights(start_w0, start_w1, start_w2)
                                if min_err < trust:
                                    self.set_weights(best_w0, best_w1, best_w2)
                                    if verbose:
                                        print "update mean square error: %f" % (min_err)
                                    return
        self.set_weights(best_w0, best_w1, best_w2)
        if verbose:
            print "update mean square error: %f" % (min_err)

    def train_N(self, indata, target, N, trust, verbose=False): # DON'T USE THIS!
        for i in xrange(N):
            self.feedforward(indata)
            self.backprop(target)
            self.train(indata, target, trust, verbose)

    def train_TOL(self, indata, target, TOL, verbose=False): # NOR THIS
        while True:
            self.feedforward(indata)
            if self.current_err_meansq(target) < TOL:
                break
            self.backprop(target)
            self.train(indata, target, trust, verbose)

    def get_weights(self):
        return self.weight0, self.weight1, self.weight2

    def set_weights(self, w0, w1, w2):
        self.weight0 = w0
        self.weight1 = w1
        self.weight2 = w2

    def gd_err(self, indata, target, step0, step1, step2):
        self.weight0 -= step0 * self.g_weight0
        self.weight1 -= step1 * self.g_weight1
        self.weight2 -= step2 * self.g_weight2
        return self.err_meansq(indata, target)

    def rand_err(self, indata, target, step0, step1, step2):
        rand_grad = lambda shape: 20 * np.random.random_sample(shape) - 10
        grad = rand_grad(self.shape_w0)
        self.weight0 -= step0 * grad
        grad = rand_grad(self.shape_w1)
        self.weight1 -= step1 * rand_grad(self.shape_w1)
        self.weight2 -= step2 * rand_grad(self.shape_w2)
        return self.err_meansq(indata, target)

    def check_grad(self):
        """apply two small perturbations to the input data, use the N-dimensional

class LSTM:
    foo = 'bar'
