#!/usr/bin/python
# _moduleFunc for module private -- will not be imported
# __classFunc for class private -- it's not actually private, but it's tradition
import numpy as np
import math
# import cython # BE SURE TO STATIC TYPE WHAT NEEDS TO BE STATIC TYPED, BRUH

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))

def _dsigmoid(x):
    return math.exp(x) / pow(1 + math.exp(x), 2)

def _dtanh(x):
    return 1 - pow(np.tanh(x), 2)

def _randND(*dims)
    return np.random.rand(dims)*2 - 1 # ayo, static type this

def _sum_sq_err(input, target):
    residue = target - input
    return 0.5 * np.dot(residue, residue)

class Net:
    """defines a neural network object"""

    bias_val = 1

    def __init__(self, input_len, output_len, layer0_len, layer1_len, layer2_len):
        """This instantiates a neural network
        
        Args:
            self (obj) -- the object being instatiated itself, implicitly called
                            WIP           
        """
        self.input_len = input_len
        self.output_len = output_len
        self.num_layers = 4
        self.layers = {
            0: np.zeros(layer0_len),
            1: np.zeros(layer1_len),
            2: np.zeros(layer2_len),
            3: np.zeros(output_len),
        }

        # add in bias term with + 1
        self.weights = {
            0: np.ones((layer0_len, input_len + 1)),
            1: np.ones((layer1_len, layer0_len + 1)),
            2: np.ones((layer2_len, layer1_len + 1)),
            3: np.ones((output_len, layer2_len + 1)),
        }

    def feedforward(input):
        """runs the net with given input and saves hidden activations"""
        if input.shape() != self.input_shape:
            raise ValueError("Input dimensions not match expected dimensions")
        work_in = input
        for i in range(self.num_layers):
            work_in = work_in.append(bias_val)
            for j in range(self.layers[i]):
                temp_val = np.dot(work_in, weights[i][j])
                self.layers[i][j] = np.tanh(temp_val)
            if i == self.num_layers - 1:
                return self.layers[i]
            else:
                work_in = self.layers[i]
        
    def train(input, target, stop):
        if target.shape() != self.output_len:
            raise ValueError("Target dimensions do not match output dimensions")    
        if 'TOL' in stop:
            # train to a certain tolerance 
            self.__trainTOL(input, target, stop['TOL'])
        elif 'N' in stop:
            # train N times
            self.__trainN(input, target, stop['N'])
        elif 'min' in stop:
            #train until error function is minimized
            self.__trainMin(input, target)
        else:
            raise TypeError("Undefined stopping condition for training")

    def __backprop(target):
        return "ayylmao"
