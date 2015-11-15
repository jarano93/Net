#!usr/bin/python

from numpy import dot
from numpy.linalg import norm
import time

class HFSMatrix():
    """
        PLS BASED GOD
    """

    # creates/opens HDF5 file
    matrix_file = h5py.File('matrix.hdf5', 'w') 

    def __init__(self, rows, cols, name, default_val=0):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.name = name
        self.dset = matrix_file.create_dataset(name, self.shape, dtype='<f8') # <f8 is code for float64
        if default_val != 0:
            for r in xrange(rows):
                for c in xrange(cols):
                    self.dset[r,c] = default_val
        file.flush()

    def __repr__(self):
        return self.name

    def __getitem__(self, key):
        return self.dset[key]

    def __setitem__(self, key, val):
        self.dset[key] = val

    @classmethod
    def add(m1, m2):
        if m1.shape != m2.shape:
            raise ValueError("Arguments need matching dimensions")
        res_name = m1.name + '_' + m2.name + '_add'
        res = HFSMatrix(m1.rows, m1.cols, res_name)
        for r in xrange(m1.rows):
            res[r,:] = m1[r,:] + m2[r,:]
        return res

    @classmethod
    def sub(m1, m2):
        if m1.shape != m2.shape:
            raise ValueError("Arguments need matching dimensions")
        res_name = m1.name + '_' + m2.name + '_sub'
        res = HFSMatrix(m1.rows, m1.cols, res_name)
        for r in xrange(m1.rows):
            res[r,:] = m1[r,:] - m2[r,:]
        return res

    @classmethod
    def neg(m):
        res_name = m.name + '_neg'
        res = HFSMatrix(m.rows, m.cols, res_name)
        for r in xrange(m.rows):
            res[r,:] = - m[r,:]
        return res

    # mul(m, arg):

    # mul_scalar(m, s, name):

    # mul_vector(m, v, name):

    # mul_matrix(m1, m2, name):

    # mul_transpose(m, t, name):

    # div(m, arg):

    # div_scalar(m, s):

    # div_matrix(m1, m2):

    
