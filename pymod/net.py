#!/usr/bin/python
# _moduleFunc for module private -- will not be imported
# __classFunc for class private -- it's not actually private, but it's tradition
import numpy as np
import numpy.linalg as la
import math
# import cython # BE SURE TO STATIC TYPE WHAT NEEDS TO BE STATIC TYPED, BRUH

# Not using sigmoid right now
# def _sigmoid(x):
#     return 1 / (1 + math.exp(-x))
#
# def _dsigmoid(x):
#     return math.exp(x) / pow(1 + math.exp(x), 2)

def _dtanh(x):
    return 1 - pow(np.tanh(x), 2)

def _ddtanh(x):
    tanh_x = np.tanh(x)
    return 2 * (pow(tanh_x, 3) - tanh_x)

def _randND(*dims):
    return np.random.rand(dims)*2 - 1 # ayo, static type this

def _sum_sq_err(input, target):
    residue = input - target
    ss_err = np.vdot(residue, residue) / 2
    if math.isnan(ss_err):
        return 1e4
    else:
        return ss_err

class Net:
    """defines a neural network object"""

    def __init__(self, input_len, output_len, layer0_len, layer1_len, layer2_len, SMD=True):
        """This instantiates a neural network
        
        Args:
            self (obj) -- the object being instatiated itself, implicitly called
                            WIP           
        """
        self.input_len = int(input_len)
        self.layer0_len = int(layer0_len)
        self.layer1_len = int(layer1_len)
        self.layer2_len = int(layer2_len)
        self.output_len = int(output_len)

        self.input = np.ones(input_len + 1)
        self.output = np.zeros(output_len)

        self.activation0 = np.zeros(layer0_len + 1)
        self.activation1 = np.zeros(layer1_len + 1)
        self.activation2 = np.zeros(layer2_len + 1)

        self.hmap0 = np.zeros(layer0_len + 1)
        self.hmap1 = np.zeros(layer1_len + 1)
        self.hmap2 = np.zeros(layer2_len + 1)

        # self.weight0 = np.ones((input_len + 1, layer0_len))
        # self.weight1 = np.ones((layer0_len + 1, layer1_len))
        # self.weight2 = np.ones((layer1_len + 1, layer2_len))
        # self.weight3 = np.ones((layer2_len + 1, output_len))

        self.weight0 = 2 * np.random.random_sample((input_len + 1, layer0_len)) - 1 
        self.weight1 = 2 * np.random.random_sample((layer0_len + 1, layer1_len)) - 1
        self.weight2 = 2 * np.random.random_sample((layer1_len + 1, layer2_len)) - 1
        self.weight3 = 2 * np.random.random_sample((layer2_len + 1, output_len)) - 1
        
        self.SMD = SMD
        self.__zero()
        if SMD:
            self.step0 = 2 * np.random.random_sample((input_len + 1, layer0_len)) - 1
            self.step1 = 2 * np.random.random_sample((layer0_len + 1, layer1_len)) - 1
            self.step2 = 2 * np.random.random_sample((layer1_len + 1, layer2_len)) - 1
            self.step3 = 2 * np.random.random_sample((layer2_len + 1, output_len)) - 1

            self.aux0 = 2 * np.random.random_sample((input_len + 1, layer0_len)) - 1
            self.aux1 = 2 * np.random.random_sample((layer0_len + 1, layer1_len)) - 1
            self.aux2 = 2 * np.random.random_sample((layer1_len + 1, layer2_len)) - 1
            self.aux3 = 2 * np.random.random_sample((layer2_len + 1, output_len)) - 1

    def feedforward(self, input): # ok
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
            self.activation2[l] = np.vdot(self.weight2[:,l], self.hmap1)
        self.hmap2 = np.tanh(self.activation2)
        # print self.hmap2
        
        for m in range(self.output_len):
            self.output[m] = np.vdot(self.weight3[:,m], self.hmap2)
            # if self.output[m] < 0:
                # self.output[m] = -1
            # else:
                # self.output[m] = 1
        # print self.output
        return self.output

    def __backprop(self, target): #  not ok
        l_product = np.zeros(self.layer2_len)
        k_product = np.zeros(self.layer1_len)
        j_product = np.zeros(self.layer0_len)

        self.delta3 = self.output - target
        for m in range(self.output_len):
            for l in range(self.layer2_len + 1):
                self.g_weight3[l,m] =  self.delta3[m] * self.hmap2[l]
        # print self.g_weight3

        for l in range(self.layer2_len):
            for m in range(self.output_len):
                self.delta2[l] += self.delta3[m] * self.weight3[l,m]
            l_product[l] = self.delta2[l] * _dtanh(self.activation2[l])
            for k in range(self.layer1_len + 1):
                self.g_weight2[k,l] =  l_product[l] * self.hmap1[k]
        # print self.g_weight2

        for k in range(self.layer1_len):
            for l in range(self.layer2_len):
                self.delta1[k] += l_product[l] * self.weight2[k,l]
            k_product[k] = self.delta1[k] * _dtanh(self.activation1[k])
            for j in range(self.layer0_len + 1):
                self.g_weight1[j,k] = k_product[k] * self.hmap0[j]
        # print self.g_weight1

        for j in range(self.layer0_len):
            for k in range(self.layer1_len):
                self.delta0[j] += k_product[k] * self.weight1[j,k]
            j_product[j] = self.delta0[j] * _dtanh(self.activation0[j])
            for i in range(self.input_len + 1):
                self.g_weight0[i,j] = j_product[j] * self.input[i]
        # print self.g_weight0

    def __r_pass(self):
        for j in range(self.layer0_len):
            for i in range(self.input_len + 1):
                self.r_activation0[j] += self.g_weight0[i,j] * self.input[i]
            self.r_hmap0[j] = _dtanh(self.activation0[j]) * self.r_activation0[j]

        for k in range(self.layer1_len):
            for j in range(self.layer0_len + 1):
                self.r_activation1[k] += self.weight1[j,k] * self.r_hmap0[j]
                self.r_activation1[k] += self.g_weight1[j,k] * self.hmap0[j]
            self.r_hmap1[k] = _dtanh(self.activation1[k]) * self.r_activation1[k]

        for l in range(self.layer2_len):
            for k in range(self.layer1_len + 1):
                self.r_activation2[l] += self.weight2[k,l] * self.r_hmap1[k]
                self.r_activation2[l] += self.g_weight2[k,l] * self.hmap1[k]
            self.r_hmap2[l] = _dtanh(self.activation2[l]) * self.r_activation2[l]

        for m in range(self.output_len):
            for l in range(self.layer2_len + 1):
                self.r_output[m] += self.weight3[l,m] * self.r_hmap2[l]
                self.r_output[m] += self.g_weight3[l,m] * self.hmap2[l]

        lead_product = 0
        tail_product = 0

        self.r_delta3 = self.r_output

        for l in range(self.layer2_len):
            for m in range(self.output_len):
                self.r_delta2[l] += self.r_delta3[m] * self.weight3[l,m]
                self.r_delta2[l] += self.delta3[m] * self.g_weight3[l,m]

        for l in range(self.layer2_len):
            lead_product = self.r_delta2[l] * _dtanh(self.activation2[l])
            lead_product += self.delta2[l] * _ddtanh(self.activation2[l]) * self.r_activation2[l]
            tail_product = self.delta2[l] * _dtanh(self.activation2[l])
            for k in range(self.layer1_len):
                self.r_delta1[k] += lead_product * self.weight2[k,l]
                self.r_delta1[k] += tail_product * self.g_weight2[k,l]

        for k in range(self.layer1_len):
            lead_product = self.r_delta1[k] * _dtanh(self.activation1[k])
            lead_product += self.delta1[k] * _ddtanh(self.activation1[k]) * self.r_activation1[k]
            tail_product = self.delta1[k] * _dtanh(self.activation1[k])
            for j in range(self.layer0_len):
                self.r_delta0[j] += lead_product * self.weight1[j,k] 
                self.r_delta0[j] += tail_product * self.g_weight1[j,k] 

        for m in range(self.output_len):
            for l in range(self.layer2_len + 1):
                self.r_weight3[l,m] = self.r_delta3[m] * self.hmap2[l]
                self.r_weight3[l,m] += self.delta3[m] * self.r_hmap2[l]

        for l in range(self.layer2_len):
            lead_product = self.r_delta2[l] * _dtanh(self.activation2[l])
            lead_product += self.delta2[l] * _ddtanh(self.activation2[l]) * self.r_activation2[l]
            tail_product = self.delta2[l] * _dtanh(self.activation2[l])
            for k in range(self.layer1_len + 1):
                self.r_weight2[k,l] = lead_product * self.hmap1[k]
                self.r_weight2[k,l] += tail_product * self.r_hmap1[k]

        for k in range(self.layer1_len):
            lead_product = self.r_delta1[k] * _dtanh(self.activation1[k])
            lead_product += self.delta1[k] * _ddtanh(self.activation1[k]) * self.r_activation1[k]
            tail_porduct = self.delta1[k] * _dtanh(self.activation1[k])
            for j in range(self.layer0_len + 1):
                self.r_weight1[j,k] = lead_product * self.hmap0[j]
                self.r_weight1[j,k] += tail_product * self.r_hmap0[j]

        for j in range(self.layer0_len):
            lead_product = self.r_delta0[j] * _dtanh(self.activation0[j])
            lead_product += self.delta0[j] * _ddtanh(self.activation0[j]) * self.r_activation0[j]
            for i in range(self.input_len + 1):
                self.r_weight0[i,j] = lead_product * self.input[i]

    def __weight_GD(self,step_size):
        self.weight0 -= step_size * self.g_weight0

        self.weight1 -= step_size * self.g_weight1

        self.weight2 -= step_size * self.g_weight2

        self.weight3 -= step_size * self.g_weight3

    def __weight_SMD(self, learn, forget): #Stochastic Meta Descent
        for i in xrange(self.input_len + 1):
            for j in xrange(self.layer0_len):
                self.step0[i,j] *= max(0.5, 1 + learn * self.aux0[i,j] * self.g_weight0[i,j])

        for j in xrange(self.layer0_len + 1):
            for k in xrange(self.layer1_len):
                self.step1[j,k] *= max(0.5, 1 + learn * self.aux1[j,k] * self.g_weight1[j,k])

        for k in xrange(self.layer1_len + 1):
            for l in xrange(self.layer2_len):
                self.step2[k,l] *= max(0.5, 1 + learn * self.aux2[k,l] * self.g_weight2[l,l])

        for l in xrange(self.layer2_len + 1):
            for m in xrange(self.output_len):
                self.step3[l,m] *= max(0.5, 1 + learn * self.aux2[l,m] * self.g_weight3[l,m])

        self.weight0 += self.step0 * self.g_weight0
        self.weight1 += self.step1 * self.g_weight1
        self.weight2 += self.step2 * self.g_weight2
        self.weight3 += self.step3 * self.g_weight3

        inner = self.g_weight0 + forget * np.dot(self.aux0, np.dot(self.g_weight0.T, self.r_weight0))
        self.aux0 = forget * self.aux0 + (self.step0 * inner)

        inner = self.g_weight1 + forget * np.dot(self.aux1, np.dot(self.g_weight1.T, self.r_weight1))
        self.aux1 = forget * self.aux1 + (self.step1 * inner)

        inner = self.g_weight2 + forget * np.dot(self.aux2, np.dot(self.g_weight2.T, self.r_weight2))
        self.aux2 = forget * self.aux2 + (self.step2 * inner)

        inner = self.g_weight3 + forget * np.dot(self.aux3, np.dot(self.g_weight3.T, self.r_weight3))
        self.aux3 = forget * self.aux3 + (self.step3 * inner)

    def __weight_CGD(self): # this is broken as hell
        denominator = np.dot(self.r_weight0.T, self.g_weight0)
        numerator = -np.dot(self.g_weight0.T, self.g_weight0)
        numerator -= np.dot(self.r_weight0.T, self.weight0)
        step = la.lstsq(denominator, numerator)[0]
        # print step
        # print "\n"
        # print self.weight0
        # print "\n"
        change = np.dot(-self.g_weight0, step)
        # print change
        # print "\n"
        self.weight0 += change
        # print self.weight0
        # raw_input("Enter...")
 
        denominator = np.dot(self.r_weight1.T, self.g_weight1)
        numerator = np.dot(self.g_weight1.T, self.g_weight1)
        numerator += np.dot(self.r_weight1.T, self.weight1)
        step = la.lstsq(denominator, numerator)[0]
        # print step
        # print "\n"
        # print self.weight1
        # print "\n"
        change = np.dot(-self.g_weight1, step)
        # print change
        # print "\n"
        self.weight1 += change
        # print self.weight1
        # raw_input("Enter...")
 
        denominator = np.dot(self.r_weight2.T, self.g_weight2)
        numerator = np.dot(self.g_weight2.T, self.g_weight2)
        numerator += np.dot(self.r_weight2.T, self.weight2)
        step = la.lstsq(denominator, numerator)[0]
        # print step
        # print "\n"
        # print self.weight2
        # print "\n"
        change = np.dot(-self.g_weight2, step)
        # print change
        self.weight2 += change
        # print self.weight2
        # print "\n"
        # raw_input("Enter...")
 
        denominator = np.dot(self.r_weight3.T, self.g_weight3)
        numerator = np.dot(self.g_weight3.T, self.g_weight3)
        numerator += np.dot(self.r_weight3.T, self.weight3)
        step = la.lstsq(denominator, numerator)[0]
        # print step
        # print "\n"
        # print self.weight3
        # print "\n"
        change = np.dot(-self.g_weight3, step)
        # print  change
        self.weight3 += change
        # print self.weight3
        # print "\n"
        # raw_input("Enter...")
 
    def __train_N(self, input, target, N, opt_func, *func_args):
        result = self.feedforward(input)
        for i in range(1, N):
            self.__backprop(target)
            if self.CGD:
                self.__r_pass()
            opt_func(*func_args)
            result = self.feedforward(input)
            self.__zero()
            print "%iteration: %i\tSumSquareError: %f" % (i, _sum_sq_err(result, target))
    
    def train_N(self, input, target, N):
        if self.SMD:
            self.__train_N_SMD(input, target, N)
        else:
            self.__train_N_GD(input, target, N)

    def err(self, input, target):
        self.feedforward(input)
        return _sum_sq_err(self.output, target)

    def train_once(self, input, target, verbose=False, **kwarg):
        result = self.feedforward(input)
        self.__backprop(target)
        if self.SMD:
            self.__r_pass()
            self.__weight_SMD(kwarg['learn'], kwarg['forget'])
        else:
            self.__weight_GD(kwarg['step_size'])
        self.feedforward(input)
        self.__zero()
        if verbose:
            print "train err: %f" % _sum_sq_err(self.output, target)

    def __train_N_GD(self, input, target, N, step_size):
        self.__train_N(input, target, N, self.__weight_GD, step_size)

    def __train_N_SMD(self, input, target, N):
        self.__train_N(input, target, N, self.__weight_SMD)

    def __zero(self):
        self.delta0 = np.zeros(self.layer0_len)
        self.delta1 = np.zeros(self.layer1_len)
        self.delta2 = np.zeros(self.layer2_len)
        self.delta3 = np.zeros(self.output_len)

        self.g_weight0 = np.zeros((self.input_len + 1, self.layer0_len))
        self.g_weight1 = np.zeros((self.layer0_len + 1, self.layer1_len))
        self.g_weight2 = np.zeros((self.layer1_len + 1, self.layer2_len))
        self.g_weight3 = np.zeros((self.layer2_len + 1, self.output_len))

        if self.SMD:
            self.r_activation0 = np.zeros(self.layer0_len + 1)
            self.r_activation1 = np.zeros(self.layer1_len + 1)
            self.r_activation2 = np.zeros(self.layer2_len + 1)

            self.r_hmap0 = np.zeros(self.layer0_len + 1)
            self.r_hmap1 = np.zeros(self.layer1_len + 1)
            self.r_hmap2 = np.zeros(self.layer2_len + 1)

            self.r_output = np.zeros(self.output_len)

            self.r_delta0 = np.zeros(self.layer0_len)
            self.r_delta1 = np.zeros(self.layer1_len)
            self.r_delta2 = np.zeros(self.layer2_len)
            self.r_delta3 = np.zeros(self.output_len)

            # a misnomer honestly, are Hessian * g_weight
            self.r_weight0 = np.zeros((self.input_len + 1, self.layer0_len)) 
            self.r_weight1 = np.zeros((self.layer0_len + 1, self.layer1_len))
            self.r_weight2 = np.zeros((self.layer1_len + 1, self.layer2_len))
            self.r_weight3 = np.zeros((self.layer2_len + 1, self.output_len))
