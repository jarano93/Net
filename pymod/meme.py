#!/usr/bin/py

import numpy as np
import math

def dtanh(x):
    return 1 - pow(np.tanh(x), 2)

def sum_sq_err(a, b):
    residue = a - b
    return np.vdot(residue, residue) / 2

def mean_sq_err(a. b):
    residue = a -b
    n = len(a)
    return np.vdot(residue, residue) / n

class MemeNet():
    """ defines a 2 layer that uses MEMES of itself to optimize itself """
    #  We Metal Gear Solid Now

    def __init__(self, input_len, output_len, layer0_len, layer1_len):
        self.input_len = int(input_len)
        self.layer0_len = int(layer0_len)
        self.layer1_len = int(layer1_len)
        self.output_len = int(output_len)

        self.input = np.ones(input_len + 1)
        self.output = np.zeros(output_len)

        self.activation0 = np.zeros(layer0_len + 1)
        self.activation1 = np.zeros(layer1_len + 1)

        self.hmap0 = np.zeros(layer0_len + 1)
        self.hmap1 = np.zeros(layer1_len + 1)

        self.shape_w0 = (input_len + 2, layer0_len)
        self.shape_w1 = (layer0_len + 1, layer1_len)
        self.shape_w2 = (layer1_len + 1, output_len)

        self.weight0 = 2 * np.random.random_sample(self.shape_w0) - 1
        self.weight1 = 2 * np.random.random_sample(self.shape_w1) - 1
        self.weight2 = 2 * np.random.random_sample(self.shape_w2) - 1

        self.g_weight0 = np.zeros(self.shape_w0)
        self.g_weight1 = np.zeros(self.shape_w1)
        self.g_weight2 = np.zeros(self.shape_w2)

    # METAL GEAR, HUH?
    def feedforward(self, input):
        """runs the net with given input and saves hidden activations"""

        local_input = np.array(input)
        if len(local_input) != self.input_len:
            raise ValueError("Input dimensions not match expected dimensions")
        
        self.input = np.append(local_input, 1)
        for j in range(self.layer0_len):
            self.activation0[j] = np.vdot(self.weight0[:,j], self.input)
        self.hmap0 = np.tanh(self.activation0)
        # print self.hmap0
        
        for k in range(self.layer1_len):
            self.activation1[k] = np.vdot(self.weight1[:,k], self.hmap0)
        self.hmap1 = np.tanh(self.activation1)
        # print self.hmap1
        
        for l in range(self.layer2_len):
            self.output[l] = np.vdot(self.weight2[:,l], self.hmap1)
        return self.output
 
    # Snake! Answer me!
    def backprop(self, target):
        k_product = np.zeros(self.layer1_len)
        j_product = np.zeros(self.layer0_len)

        delta2 = self.output - target
        for l in xrange(self.output_len):
            for k in xrange(self.layer1_len + 1):
                self.g_weight2[k,l] = delta2[l] * self.hmap1[k]

        delta1 = np.zeros(self.layer1_len)
        for k in xrange(self.layer1_len);
            for  l in xrange(self.output_len):
                delta1[k] += delta2[l] * self.hmap1[k]
            k_product[k] = delta1[l] * dtanh(self.activation1[k])
            for j in xrange(self.layer0_len + 1):
                self.g_weight1[j,k] = k_product[k] * self.hmap0[j]

        delta0 = np.zeros(self.layer0_len)
        for j in range(self.layer0_len):
            for k in range(self.layer1_len):
                delta0[j] += k_product[k] * self.weight1[j,k]
            j_product[j] = delta0[j] * _dtanh(self.activation0[j])
            for i in range(self.input_len + 1):
                self.g_weight0[i,j] = j_product[j] * self.input[i]

    # Snake?
    def result(self):
        return self.output

    # Snake!?
    def err_sumsq(self, data, target):
        self.feedforward(data)
        return sum_sq_err(self.output, target)

    # SNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKE!
    def err_meansq(self, data):
        self.feedforward(data)
        return mean_sq_err(self.output, target)

    # Snake, try to remeber the basics of CQC
    def curent_err_sumsq(self, target):
        return sum_sq_err(self.output, target)

    
    def current_err_meansq(self, target):
        return mean_sq_err(self.output, target)

    def train(self, input, target, verbose=False):
        # The time has come to an end
        start_w0 = self.weight0
        start_w0 = self.weight0
        start_w0 = self.weight0
        # Ye-ah this is what nature planned
        best_w0 = self.weight0
        best_w1 = self.weight1
        best_w2 = self.weight2
        # Being tracked by a starving beast
        min_err = self.current_err_meansq()
        # Looking for its daily feast
        mags = np.linspace(-2, -6, 5)
        # A predator on the verge of death
        mags = np.power(10, mags)
        # Close to its last breath!
        leads = np.linspace(1, 9.5, 18)
        # Getting close to its last breath!

        # RULES!
        # OF!
        # NATURE!
        for l0 in leads:
            for l1 in leads:
                for l2 in leads:
                    for m0 in mags:
                        for m1 in mags:
                            for m2 in mags:
                                err_gd = self.gd_err(input, target, l0*m0, l1*m1, l2*m2)
                                if err < min_err:
                                    min_err = err_gd
                                    best_w0, best_w1, best_w2 = self.get_weights()
                                self.set_weights(start_w0, start_w1, start_w2)
                                for i in xrange(10):
                                    err_rand = self.rand_err(input, target, l0*m0, l1*m1, l2*m2)
                                    if err_rand < min_err:
                                        min_err = err_rand
                                        best_w0 = best_w1, best_w2 = self.get_weights()
                                    self.set_weights(start_w0, start_w1, start_w2)
        self.set_weights(best_w0, best_w1, best_w2)
        if verbose:
            print min_err

    def train_N(self, input, target, N, verbose=False):
        for i in xrange(N):
            self.feedforward(input)
            self.backprop(target)
            self.train(input, target, verbose)

    def train_TOL(self, input, target, TOL, verbose=False):
        while True:
            self.feedforward(input)
            if self.current_err_meansq(target) < TOL:
                break
            self.backprop(target)
            self.train(input, target, verbose)

    def get_weights(self):
        return self.weight0, self,weight1, self.weight2

    def set_weights(self, w0, w1, w2):
        self.weight0 = w0
        self.weight1 = w1
        self.weight2 = w2

    def gd_err(self, input, target, step0, step1, step2):
        self.weight0 -= step0 * self.g_weight0
        self.weight1 -= step1 * self.g_weight1
        self.weight2 -= step2 * self.g_weight2
        return self.err_meansq(self, input, target)

    def rand_err(self, input, target, step0, step1, step2):
        rand_grad = lambda shape: 20 * np.random.random_sample(shape) - 10
        self.weight0 -= step0 * rand_grad(self.shape_w0)
        self.weight1 -= step1 * rand_grad(self.shape_w1)
        self.weight2 -= step2 * rand_grad(self.shape_w2)
        return self.err_meansq(self, input, target)

    # [OCELOT HISSING INTENSIFIES IN THE DISTANCE]
