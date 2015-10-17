#!/usr/bin/python
# _moduleFunc for module private -- will not be imported
# __classFunc for class private -- it's not actually private, but it's tradition
import numpy as np
import numpy.linalg as la
# import cython # YEE BRAH, U FINNA USE CYTHON
# import mod_hfo as hfo

def cgd(A, x, b, tol):
    """Implementation of conjugate gradient descent"""
    res = b - np.dot(A, x)
    p = res
    res_norm_old = la.norm(res) 
    while True:
        Ap = np.dot(A, p)
        alpha = np.dot(res_norm_old) / np.dot(p, Ap)
        x = x + np.dot(alpha, p)
        res = res - alpha * Ap
        res_norm_new = la.norm(res)
        if res_norm_new < tol:
            break
        p = r + (res_norm_new / res_norm_old) * p
        res_norm_old = res_norm_new
    return x

"""
def pcg():
    return 0
"""

def sgd():
    return 0

def hfo():
    return 0
