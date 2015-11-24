#!usr/bin/python

import numpy as np

def pca(data, num):
    # at the moment can only handle a data matrix of N observations each with D length
    # so data must be 2D
    # assumed the observations are store along the 0th axis of data
    zero_mean = data - np.mean(data, axis=0)
    U, S = np.linalg.svd(zero_mean)[0:2]
    S_diag = np.zeros((data.shape[0], num))
    for i in xrange(num):
        S_diag[i,i] = S[i]
    return np.dot(U, S_diag)

# idk how practical MiniBatch and WideBatch are in retrospect, but at the time
# I wrote them they seemed very good
class MiniBatch():
    """
        given data where dim(data) >= 2 delivers batches piecewise
        data must also have shape attribute
        will only subdivide the last two dimensions of the data
    """

    def __init__(self, data, batch_shape):
        if len(data) < 2:
            raise ValueError("data needs atleast two dimensions")
        if len(batch_shape) < 2:
            raise ValueError("batch_shape needs atleast two dimensions")
        for i in range(2):
            if data.shape[-2 + i] % batch_shape[i] != 0:
                raise ValueError("batch_shape needs to cleanly divide the data the last two dimensions of the data")
        self.batch_shape = batch_shape
        self.batch_rows = batch_shape[0]
        self.batch_cols = batch_shape[1]
        self.data = data
        self.data_rows = data.shape[-2] / batch_shape[0]
        self.data_cols = data.shape[-1] / batch_shape[1]

    def __getitem__(self, key):
        """ No slices """
        if len(key) != 2:
            raise ValueError("Can only subdivide by the last two dimensions of the data!")
        (r, c) = key
        row_slice = slice(r * batch_rows, (r + 1) * batch_rows)
        col_slice = slice(r * batch_cols, (r + 1) * batch_cols)
        return self.data[...,row_slice,col_slice]

class WideBatch(MiniBatch):
    def __init__(self, data, batch_shape):
        MiniBatch.__init__(self, data, batch_shape)
        self.num_batches = self.data_rows * self.data_cols
        data_shape = data.shape
        pad_shape = data_shape
        pad_shape[-2] = batch_shape[0]
        pad_shape[-1] = batch_shape[1]
        self.pad = np.zeros(pad_shape)

    def __getitem__(self,key):
        """ No slices """
        if len(key) != 2:
            raise ValueError("Can only subdivide by the last two dimensions of the data!")
        (r, c) = key
        row_slice = slice(r * batch_rows, (r + 1) * batch_rows)
        col_slice = slice(c * batch_cols, (c + 1) * batch_cols)
        bottom_right =  self.data[..., row_slice, col_slice]
        (r_back, c_back) = (r - 1, c - 1)
        row_back_slice = slice(row_back * batch_rows, r * batch_rows)
        col_back_slice = slice(col_back * batch_cols, c * batch_cols)
        if r_back < 0:
            top_left = self.pad
            bottom_left = self.pad
        else:
            bottom_left = self.data[..., row_back_slice, col_slice]
        if c_back < 0:
            top_left = self.pad
            top_right = self.pad
        else:
            top_right = self.data[..., row_slice, col_back_slice]
        if r_back > 0 and c_back > 0:
            top_left = self.data[..., row_back_slice, col_back_slice]
        left = np.append(top_left, bottom_left, axis=-2)
        right = np.append(top_right, bottom_right, axis=-2)
        return np.append(left, right, axis=-1)
