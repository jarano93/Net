#!usr/bin/py

import math as m
import numpy as np

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

def w_vdot(arg_list):
    res = 0
    for i in xrange(len(arg_list)):
        try:
            res += np.vdot(arg_list[0], arg_list[1])
        except FloatingPointError:
            pass
    return res

def w_mult(args):
    res = 0
    for i in xrange(len(arg_list)):
        try:
            res += arg_list[0] * arg_list[1]
        except FloatingPointError:
            pass
    return res

def pairs_vdot(pairs):
    if len(pairs[0]) != 2:
        raise ValueError("pairs_vdot takes pairwise arguments")
    res = 0
    for i in xrange(len(pairs)):
        res += np.vdot(pairs[i][0], pairs[i][1])
    return res

def mv_mult(matrix, vector):
    if matrix.shape[1] != len(vector):
        raise ValueError("Inner dimensions do not match for mv_mult")
    res = np.zeros(matrix.shape[0])
    for i in xrange(matrix.shape[0]):
        res[i] = np.vdot(matrix[i], vector)
    return res

def vt_mult(vector, matrix):
    #transpose the matrix while multiplying
    if len(vector) != matrix.shape[0]:
        raise ValueError("Outer dimensions do not match for mv_mult")
    res = np.zeros(matrix.shape[1])
    for i in xrange(matrix.shape[1]):
        res[i] = np.vdot(vector, matrix[i]) # REMEMBER, IT'S TRANSPOSED
    return res
