#!usr/bin/python

import numpy as np
import numpy.linalg as la

def sum_sq_err(a, b):
    residue = a - b
    return np.vdot(residue, residue) / 2

class FastNet():
    """defines an 2 layer extreme learning machine"""

    def __init__(self, input_len, output_len, layer0_len, layer1_len):
        self.input_len = int(input_len)
        self.layer0_len = int(layer0_len)
        self.layer1_len = int(layer1_len)
        self.output_len = int(output_len)

        self.output = np.zeros(output_len)

        self.weight0 = 2 * np.random.random_sample((layer0_len, input_len + 1)) - 1
        self.weight1 = 2 * np.random.random_sample((layer1_len, layer0_len + 1)) - 1
        self.weight2 = 2 * np.random.random_sample((output_len, layer1_len + 1)) - 1
        
        self.activation = np.zeros(layer1_len)

        self.err_length = 10
        self.errs = np.zeros(self.err_length)
        self.judge = False
        self.err_num = 0
        self.jump = False

    def result(self):
        return self.output

    def run(self, input):
        local_input = np.array(input)
        if len(local_input) != self.input_len:
            raise ValueError("Input length does not match expected length")

        local_input = np.append(local_input, 1)

        work_val1 = np.ones(self.layer0_len + 1)
        for j in xrange(self.layer0_len ):
            work_val1[j] = np.vdot(self.weight0[j,:], local_input)
        work_val1 = np.tanh(work_val1)

        work_val2 = np.ones(self.layer1_len + 1)
        for k in xrange(self.layer1_len):
            work_val2[k] = np.vdot(self.weight1[k,:], work_val1)
        work_val2 = np.tanh(work_val2)
        self.activation = work_val1

        for l in xrange(self.output_len):
            self.output[l] = np.vdot(self.weight2[l,:], work_val2)
        self.output = np.tanh(self.output)

        return self.output

    def get_grad(self, target):
        # if len(target) != self.output_len:
            # raise ValueError("Target length does not match expected length")
        residue = self.output - target
        gradient = np.zeros((self.output_len, self.layer1_len + 1))
        for l in xrange(self.output_len):
            for k in xrange(self.layer1_len + 1):
                gradient[l,k] = residue[l] * self.activation[k]
        return gradient

    def err(self, data, target):
        self.run(data)
        return sum_sq_err(self.output, target)

    def cerr(self, target):
        return sum_sq_err(self.output, target)

    def train(self, data, target, step_size, verbose=False, single=True):
        self.run(data)
        # if single:
            # self.update_err(target)
        self.update_weight2(target, step_size)
        if verbose:
            print "%f" % self.err(data, target)

    def train_N(self, data, target, step_size, N, verbose=False):
        for i in xrange(N):
            # self.update_err(target)
            self.train(data, target, step_size, verbose, False)

    def update_err(self, target):
        self.errs[self.err_num] = self.cerr(target)
        self.err_num = (self.err_num + 1) %  self.err_length
        if self.err_num == 5:
            self.judge = True
        if self.judge:
            self.jump_step()

    def update_weight2(self, target, step_size):
        # if self.jump:
            # rand_grad = np.random.random_sample((output_len, layer1_len - 0.5))
            # self.weight2 -= self.step_size * rand_grad
        # else:
        self.weight2 -= step_size * self.get_grad(target)

    def jump_step(self):
        threshold = 1e-5
        cyclic_thrshold = 1e1

        n = self.err_num
        errs = self.errs

        prev_ratio = lambda x: (errs[n] - errs[n - x]) / 2
        prev_ratios = lambda: (prev_ratio(1) + prev_ratio(2)) / 2
        if abs(self.errs[n]) < threshold:
            jump = False
        else:
            if prev_ratios() > 10:
                # print "jump"
                jump = True
                self.step_size *= 4/3
            elif prev_ratios() >= -1:
                # increase step size
                # print "grow"
                self.step_size *= 3/2
            else:
                # decrease step_size
                # print "shrink"
                self.step_size *= 2/3
