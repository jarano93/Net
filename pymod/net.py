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
    return 0.5 * np.dot(residue, residue)

class Net:
    """defines a neural network object"""

    def __init__(self, input_len, output_len, layer0_len, layer1_len, layer2_len, CGD=True):
        """This instantiates a neural network
        
        Args:
            self (obj) -- the object being instatiated itself, implicitly called
                            WIP           
        """
        self.input_len = input_len
        self.layer0_len = layer0_len
        self.layer1_len = layer1_len
        self.layer2_len = layer2_len
        self.output_len = output_len

        self.input = np.ones(input_len + 1)
        self.output = np.zeros(output_len)

        self.activation0 = np.ones(layer0_len + 1)
        self.activation1 = np.ones(layer1_len + 1)
        self.activation2 = np.ones(layer2_len + 1)

        self.hmap0 = np.zeros(layer0_len + 1)
        self.hmap1 = np.zeros(layer1_len + 1)
        self.hmap2 = np.zeros(layer2_len + 1)

        self.weight0 = np.ones((input_len + 1, layer0_len))
        self.weight1 = np.ones((layer0_len + 1, layer1_len))
        self.weight2 = np.ones((layer1_len + 1, layer2_len))
        self.weight3 = np.ones((layer2_len + 1, output_len))
        
        self.CGD = CGD
        self.__zero()

    def feedforward(self, input): # ok
        """runs the net with given input and saves hidden activations"""

        local_input = np.array(input)
        if len(local_input) != self.input_len:
            raise ValueError("Input dimensions not match expected dimensions")
        
        self.input = np.append(local_input, 1)
        for j in range(self.layer0_len):
            self.activation0[j] = np.vdot(self.weight0[:,j], self.input)
        self.hmap0 = np.tanh(self.activation0)
        
        for k in range(self.layer1_len):
            self.activation1[k] = np.vdot(self.weight1[:,k], self.hmap0)
        self.hmap1 = np.tanh(self.activation1)
        
        for l in range(self.layer2_len):
            self.activation2[l] = np.vdot(self.weight2[:,l], self.hmap1)
        self.hmap2 - np.tanh(self.activation2)

        for m in range(self.output_len):
            self.output[m] = np.vdot(self.weight3[:,m], self.hmap2)

        return self.output

    def __backprop(self, target): # ok
        l_product = np.zeros(self.layer2_len)
        k_product = np.zeros(self.layer1_len)
        j_product = np.zeros(self.layer0_len)

        self.delta3 = self.output - target
        for m in range(self.output_len):
            for l in range(self.layer2_len + 1):
                self.g_weight3[l,m] =  self.delta3[m] * self.hmap2[l]
        for l in range(self.layer2_len):
            l_product[l] = self.delta2[l] * _dtanh(self.activation2[l])
            for m in range(self.output_len):
                self.delta2[l] += self.delta3[m] * self.weight3[l,m]
            for k in range(self.layer1_len + 1):
                self.g_weight2[k,l] =  l_product[l] * self.hmap1[k]
        for k in range(self.layer1_len):
            k_product[k] = self.delta1[k] * _dtanh(self.activation1[k])
            for l in range(self.layer2_len):
                self.delta1[k] += l_product[l] * self.weight2[k,l]
            for j in range(self.layer1_len + 1):
                self.g_weight1[j,k] = k_product[k] * self.hmap0[j]
        for j in range(self.layer0_len):
            j_product[j] = self.delta0[j] * _dtanh(self.activation0[j])
            for k in range(self.layer1_len):
                self.delta0[j] += k_product[k] * self.weight1[j,k]
            for i in range(self.input_len + 1):
                self.g_weight0[i,j] = j_product[j] * self.input[i])

    def __r_pass(self):
        for j in range(self.layer0_len):
            for i in range(self.input_len + 1):
                self.r_activation0[j] += self.g_weight0[i,j] * self.input[i]
            self.r_hmap0[j] = _dtanh(self.activation0[j]) * self.r_activation[j]

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
                self.r_weight2[k,l] = lead_product * self.hmap2[k]
                self.r_weight2[k,l] += tail_product * self.r_hmap2[k]

        for k in range(self.layer1_len):
            lead_product = self.r_delta1[k] * _dtanh(self.activation1[k])
            lead_prodcut += self.delta1[k] * _ddtanh(self.activation1[k]) * self.r_activation1[k]
            tail_porduct = self.delta1[k] * _dtanh*self.activation1[k])
            for j in range(self.layer0_len + 1):
                self.r_weight1[j,k] = lead_product * self.hmap1[j]
                self.r_weight1[j,k] += tail_product * self.r_hmap1[j]

        for j in range(self.layer0_len):
            lead_product = self.r_delta0[j] * _dtanh(self.activation0[j])
            lead_product += self.delta0[j] * _ddtanh(self.activation0[j]_ * self.r_activation0[j]
            for i in range(self.input_len):
                self.r_weight0[i,j] = lead_product * self.input[i]

    def __weight_GD(step_size):
        direction0 = -self.g_weight0
        self.weight0 += step_size * direction0

        direction1 = -self.g_weight1
        self.weight1 += step_size * direction1

        direction2 = -self.g_weight2
        self.weight2 += step_size * direction2

        direction3 = -self.g_weight3
        self.weight3 += step_size * direction3

    def __weight_CGD():
        direction0 = -self.g_weight0
        denom0 = self.r_weight0 % direction0
        numer0 = direction0 % self.g_weight0
        numer0 += self.r_weight0 % -self.weight0
        step0 = numer0 / denom0
        self.weight0 += direction0 * step0
 
        direction1 = -self.g_weight1
        denom1 = self.r_weight1 % direction1
        numer1 = direction1 % self.g_weight1
        numer1 += self.r_weight1 % -self.weight1
        step1 = numer1 / denom1
        self.weight1 += direction1 * step1
 
        direction2 = -self.g_weight2
        denom2 = self.r_weight2 % direction2
        numer2 = direction2 % self.g_weight2
        numer2 += self.r_weight2 % -self.weight2
        step2 = numer2 / denom2
        self.weight2 += direction2 * step2
 
        direction3 = -self.g_weight3
        denom3 = self.r_weight3 % direction3
        numer3 = direction3 % self.g_weight3
        numer3 += self.r_weight3 % -self.weight3
        step3 = numer3 / denom3
        self.weight3 += direction3 * step3
 
    def __train_N(self, input, target, N, opt_func, *func_args):
        result = self.feedforward(input)
        for i in range(N - 1):
            self.__backprop(target)
            if self.CGD:
                self.__r_pass()
            opt_func(*func_args)
            result = self.feedforward(input)
            self.__zero()
            print _sum_sq_err(result, target)
        return result
    
    def train_N(self, input, target, N):
        if self.CGD:
            return self.__train_N_CGD(input, target, N)
        else:
            return self.__train_N_GD(input, target, N)

    def __train_N_GD(self, input, target, N, step_size):
        self.__train_N(input, target, N, self.__weight_GD, step_size)

    def __train_N_CGD(self, input, target, N):
        self.__train_N(input, target, N, self.__weight_CGD)
            for k in range(self.layer1_len):

    def __zero(self):
        self.delta0 = np.zeros(layer0_len)
        self.delta1 = np.zeros(layer1_len)
        self.delta2 = np.zeros(layer2_len)
        self.delta3 = np.zeros(output_len)

        self.g_weight0 = np.zeros((input_len + 1, layer0_len))
        self.g_weight1 = np.zeros((layer0_len + 1, layer1_len))
        self.g_weight2 = np.zeros((layer1_len + 1, layer2_len))
        self.g_weight3 = np.zeros((layer2_len + 1, output_len))
        if self.CGD:
            self.r_activation0 = np.zeros(layer0_len + 1)
            self.r_activation1 = np.zeros(layer1_len + 1)
            self.r_activation2 = np.zeros(layer2_len + 1)

            self.r_hmap0 = np.zeros(layer0_len)
            self.r_hmap1 = np.zeros(layer1_len)
            self.r_hmap2 = np.zeros(layer2_len)

            self.r_output = np.zeros(output_len)

            self.r_delta0 = np.zeros(layer0_len)
            self.r_delta1 = np.zeros(layer1_len)
            self.r_delta2 = np.zeros(layer2_len)
            self.r_delta3 = np.zeros(layer3_len)

            self.r_weight0 = np.zeros((input_len + 1, layer0_len))
            self.r_weight1 = np.zeros((layer0_len + 1, layer1_len))
            self.r_weight2 = np.zeros((layer1_len + 1, layer2_len))
            self.r_weight3 = np.zeros((layer2_len + 1, output_len))
