#!/usr/bin/py

import numpy as np
import math

def eye(x):
    return x

def deye(x):
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

def sum_sq_err(a, b):
    residue = a - b
    return np.vdot(residue, residue) / 2

def mean_sq_err(a, b):
    n = len(b)
    if len(a) != len(b):
        raise ValueError("inputs need same length")
    residue = a - b
    return np.vdot(residue, residue) / n

class MemeFFNN():
    """ defines a 2 layer that uses MEMES of itself to optimize itself """
    #  We Metal Gear Solid Now

    def __init__(self, indata_len, output_len, layer0_len, layer1_len):
        self.indata_len = int(indata_len)
        self.layer0_len = int(layer0_len)
        self.layer1_len = int(layer1_len)
        self.output_len = int(output_len)

        self.indata = np.ones(indata_len + 1) #, dtype='f')
        self.output = np.zeros(output_len) #, dtype='f')

        self.activation0 = np.zeros(layer0_len + 1) #, dtype='f')
        self.activation1 = np.zeros(layer1_len + 1) #, dtype='f')
        self.activation2 = np.zeros(output_len) #, dtype='f')

        self.hmap0 = np.zeros(layer0_len + 1) #, dtype='f')
        self.hmap1 = np.zeros(layer1_len + 1) #, dtype='f')

        self.shape_w0 = (indata_len + 1, layer0_len)
        self.shape_w1 = (layer0_len + 1, layer1_len)
        self.shape_w2 = (layer1_len + 1, output_len)

        self.weight0 = 2 * np.random.random_sample(self.shape_w0) - 1
        self.weight1 = 2 * np.random.random_sample(self.shape_w1) - 1
        self.weight2 = 2 * np.random.random_sample(self.shape_w2) - 1

    # METAL GEAR, HUH?
    def feedforward(self, indata, output=True):
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
            self.output[l] = np.vdot(self.weight2[:,l], self.hmap1)
        if output:
            return self.output

    def forgetforward(self, indata):
        """runs the net with given indata and doesn't save hidden activations"""
        if len(indata) != self.indata_len:
            print self.indata_len
            print len(indata)
            raise ValueError("Input dimensions not match expected dimensions")
        
        local_data = np.append(indata, 1)
        activation1 = np.ones(self.layer0_len + 1)
        for j in range(self.layer0_len):
            activation1[j] = np.vdot(self.weight0[:,j], local_data)
        activation1 = np.tanh(activation1)
        
        activation2 = np.ones(self.layer1_len + 1)
        for k in range(self.layer1_len):
            activation2[k] = np.vdot(self.weight1[:,k], activation1)
        activation2 = np.tanh(activation2)
        
        activation1 = np.ones(self.output_len)
        for l in range(self.output_len):
            activation1[l] = np.vdot(self.weight2[:,l], activation2)
        return activation1
 
    # Snake! Answer me!
    def backprop(self, target):
        np.seterr(all='raise')

        grad0 = np.zeros(self.shape_w0)
        grad1 = np.zeros(self.shape_w1)
        grad2 = np.zeros(self.shape_w2)

        k_product = np.zeros(self.layer1_len)
        j_product = np.zeros(self.layer0_len)
        target_array = np.array(target)

        n = np.copy(self.output_len)

        # delta2 = 2 * np.absolute(self.output - np.array(target))  / self.output_len
        delta2 = np.zeros(self.output_len)
        for l in xrange(self.output_len):
            # try:
            delta2[l] = 2 * abs(self.output[l] - target_array[l]) / n
            # except:
                # print delta2[l]
                # print self.output[l]
                # print target_array[l]
                # print n
                # raw_input('hold')
            if delta2[l] < 1e-7:
                delta2[l] = 0
            for k in xrange(self.layer1_len + 1):
                grad2[k,l] = delta2[l] * self.hmap1[k]
                if grad2[k,l] < 1e-8:
                    grad2[k,l] = 0

        delta1 = np.zeros(self.layer1_len)
        for k in xrange(self.layer1_len):
            for  l in xrange(self.output_len):
                delta1[k] += delta2[l] * self.hmap1[k]
            if delta1[k] < 1e-7:
                delta1[k] = 0
            k_product[k] = delta1[l] * dtanh(self.activation1[k])
            for j in xrange(self.layer0_len + 1):
                grad1[j,k] = k_product[k] * self.hmap0[j]
                if grad1[j,k] < 1e-7:
                    grad1[j,k] = 0

        delta0 = np.zeros(self.layer0_len)
        for j in range(self.layer0_len):
            for k in range(self.layer1_len):
                delta0[j] += k_product[k] * self.weight1[j,k]
            if delta0[j] < 1e-7:
                delta0[j] = 0
            j_product[j] = delta0[j] * dtanh(self.activation0[j])
            for i in range(self.indata_len + 1):
                grad0[i,j] = j_product[j] * self.indata[i]
                if grad0[i,j] < 1e-7:
                    grad0[i,j] = 0

        del delta0, delta1, delta2, k_product, j_product, target_array
        return grad0, grad1, grad2

    # Snake?
    def result(self):
        return self.output

    # Snake!?
    def err_sumsq(self, data, target):
        return sum_sq_err(self.forgetforward(data), target)

    # SNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKE!
    def err_meansq(self, data, target):
        return mean_sq_err(self.forgetforward(data), np.array(target))
 
    def set_meansq(self, dataset, targetset):
        if len(dataset) != len(targetset):
            raise ValueError("Need equal matching length sets!")
        n = len(dataset)
        output = np.zeros(targetset.shape)
        errs = []
        for i in xrange(n):
            output[i] = self.forgetforward(dataset[i,:])
            errs.append(mean_sq_err(output[i], targetset[i]))
        err_val = np.mean(errs)
        del output, errs
        return err_val

    # Snake, try to remeber the basics of CQC
    def curent_err_sumsq(self, target):
        return sum_sq_err(self.output, target)

    def current_err_meansq(self, target):
        return mean_sq_err(self.output, target)

    # only use this if a single observation is representative of the entire dataset
    def train(self, indata, target, trust, verbose=False):
        # The time has come to an end
        start_w0, start_w1, start_w2 = self.get_weights()
        # Ye-ah this is what nature planned
        best_w0 , best_w1 , best_w2 = start_w0, start_w1, start_w2
        # Being tracked by a starving beast
        min_err = self.current_err_meansq(target)
        if min_err < trust:
            if verbose:
                print "update mean square error: %f" % (min_err)
            return
        # Looking for its daily feast
        mags = np.linspace(-6, -2, 5)
        # A predator on the verge of death
        mags = np.power(10, mags)
        # Close to its last breath!
        leads = np.linspace(1, 5, 2)
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

    def train_set(self, trainset, targetset, trust, verbose=False, rands=20):
        # now with more copypasting
        start_w0, start_w1, start_w2 = self.get_weights()
        best_w0 , best_w1 , best_w2 = self.get_weights()

        min_err = self.set_meansq(trainset, targetset)
        if min_err < trust:
            if verbose:
                print "set mean square error: %f" % (min_err)
            return

        mags = np.linspace(-2, -5, 3)
        mags = np.power(10, mags)
        leads = np.linspace(1, 5, 2)
        steps = np.outer(leads, mags).flatten()

        local_err = 0
        for i in xrange(trainset.shape[0]):
            cand0, cand1, cand2 = self.grad_candidates(trainset[i], targetset[i])
            for s0 in steps:
                for s1 in steps:
                    for s2 in steps:
                            self.update_weights(s0, cand0, s1, cand1, s2, cand2)
                            local_err = self.set_meansq(trainset, targetset)
                            if local_err < trust:
                                min_err = local_err
                                if verbose:
                                    print "set mean square error: %f" % (min_err)
                                return
                            if local_err < min_err:
                                min_err = local_err
                                best_w0, best_w1, best_w2 = self.get_weights()
                            self.set_weights(start_w0, start_w1, start_w2)

        steps = steps / 10
        for i in xrange(rands):
            cand0, cand1, cand2 = self.rand_candidates()
            for s0 in steps:
                for s1 in steps:
                    for s2 in steps:
                            self.update_weights(s0, cand0, s1, cand1, s2, cand2)
                            local_err = self.set_meansq(trainset, targetset)
                            if local_err < trust:
                                min_err = local_err
                                if verbose:
                                    print "set mean square error: %f" % (min_err)
                                return
                            if local_err < min_err:
                                min_err = local_err
                                best_w0, best_w1, best_w2 = self.get_weights()
                            self.set_weights(start_w0, start_w1, start_w2)
        self.set_weights(best_w0, best_w1, best_w2)
        if verbose:
            print "set mean square error: %f" % (min_err)

    def train_set_N(self, trainset, targetset, trust, N, verbose=False, rands=20):
        for n in xrange(N):
            self.train_set(trainset, targetset, trust, verbose, rands=20)

    def train_set_TOL(self, trainset, targetset, TOL, verbose=False, rands=20):
        err = self.set_meansq(trainset, targetset)
        while True:
            if err < TOL:
                break
            self.train_set(trainset, targetset, TOL, verbose, rands)
            err = self.set_meansq(trainset, targetset)

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
        return np.copy(self.weight0), np.copy(self.weight1), np.copy(self.weight2)

    def set_weights(self, w0, w1, w2):
        self.weight0 = np.copy(w0)
        self.weight1 = np.copy(w1)
        self.weight2 = np.copy(w2)

    def grad_candidates(self, indata, target):
        self.feedforward(indata, False)
        return self.backprop(target)

    def rand_candidates(self):
        rand_grad = lambda shape: 2 * np.random.random_sample(shape) - 1
        rand0 = rand_grad(self.shape_w0)
        rand1 = rand_grad(self.shape_w1)
        rand2 = rand_grad(self.shape_w2)
        return rand0, rand1, rand2
    
    def update_weights(self, s0, w0, s1, w1, s2, w2):
        self.weight0 -= s0 * w0
        self.weight1 -= s1 * w1
        self.weight2 -= s2 * w2

    def grad_descent(self, indata, target, step0, step1, step2):
        self.feedforward(indata, False)
        grad0, grad1, grad2 = self.backprop(target)
        self.weight0 -= step0 * grad0
        self.weight1 -= step1 * grad1
        self.weight2 -= step2 * grad2

    def gd_err(self, indata, target, step0, step1, step2):
        grad0, grad1, grad2 = self.backprop(target)
        self.weight0 -= step0 * self.g_weight0
        self.weight1 -= step1 * self.g_weight1
        self.weight2 -= step2 * self.g_weight2
        return self.err_meansq(indata, target)

    def set_gd_err(self, trainset, targetset, step0, step1, step2):
        self.weight0 -= step0 * self.g_weight0
        self.weight1 -= step1 * self.g_weight1
        self.weight2 -= step2 * self.g_weight2
        return self.set_meansq(trainset, targetset)

    def set_rand_err(self, trainset, targetset, step0, step1, step2):
        rand_grad = lambda shape: 20 * np.random.random_sample(shape) - 10
        grad = rand_grad(self.shape_w0)
        self.weight0 -= step0 * grad
        grad = rand_grad(self.shape_w1)
        self.weight1 -= step1 * rand_grad(self.shape_w1)
        self.weight2 -= step2 * rand_grad(self.shape_w2)
        return self.set_meansq(trainset, targetset)

    # [OCELOT HISSING INTENSIFIES IN THE DISTANCE]
