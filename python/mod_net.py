#!/usr/bin/python
# _moduleFunc for module private -- will not be imported
# __classFunc for class private -- it's not actually private, but it's tradition
import numpy as np
import math
import cython # BE SURE TO STATIC TYPE WHAT NEEDS TO BE STATIC TYPED, BRUH
import mod_pcg as pcg

def _randND(*dims)
    return np.random.rand(dims)*2 - 1 # ayo, static type this

def _norm(ndArray)
    dims = ndArray.shape()
    dims_length = len(dims)
    axes_list = range(dims_length)
    return np.tensordot(ndArray, ndArray, axes=(axes_list, axes_list))

class Net:
    """defines a neural network object"""
    def __init__(self, input_shape, output_shape, layer_data):
        """This instantiates a neural network
        
        Args:
            self (obj) -- the object being instatiated itself, implicitly called
            input_shape (tuple) -- defines the dimensions of the input data
            output_shape (tuple) -- defines the dimensions of the output data
            layer_data (tuple) -- sets the number of layers and nodes per layer
                number of layers is number of size of the tuple
                each individual elem defines the number of nodes in that layer
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_weights = {}
        layer_data_len = len(layer_data)
        if layer_data_len < 1:
            raise ValueError("layer_data needs an element")
        for i in range(0,layer_data_len):
            if type(layer_data[i]) != int:
                raise TypeError("layer_data must be a tuple of ints")
            if layer_data[i] < 1:
                raise ValueError("Each layer needs atleast one node")
        self.num_weight_sets = layer_data_len + 1
        #BE SURE TO STATIC TYPE THIS BITCH FOR CYTHON
        temp_weights
        temp_shape
        for i in range(0, layer_data_len):
            if i == 0:
                # set first hidden node layer weights to fit input_shape
                temp_shape = (layer_data[0]) + input_shape
            else:
                temp_shape = (layer_data[i], layer_data[i-1])
            temp_weights = _randND(*temp_shape)
            self.hidden_weights.update({i: temp_weights})
            if i == layer_data_len - 1:
                temp_shape = output_shape + (layer_data[i])
                self.hidden_weights.update({i+1: _randND(*temp_shape)})

    def feedforward(input):
        if input.shape() != self.input_shape:
            raise ValueError("Input dimensions not match expected dimensions")
        
    def train(input, target, stop):
        if target.shape() != self.output_shape:
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

    def __trainTOL(input, target, TOL):
        err = self.__percentErr(input, target)
        while err < TOL:
            self.__backprop(target)
            self.__percentErr(input, target)

    def __trainN(input, target, N):
        for i in range(N):
            self.__backprop(target)

    def __trainMin(input, target):
        #minimize __sum_sq_err

    def __percentErr(input, target):
        #frobenius (L2) norm used to calculate error
        abs_error = _norm(self.feedforward(input) - target)
        return abs_error / _norm(target)

    def __sum_sq_err(input, target):
        return pow(_norm(self.feedforward(input) - target), 2) / 2

    def __backprop(target):
        return "ayylmao"

    def __sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __dsigmoid(x):
        return math.exp(x) / pow(1 + math.exp(x), 2)
