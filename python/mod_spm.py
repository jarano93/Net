#!/usr/bin/python

import numpy as np

class Sparse():

    def __init__(self, rows, cols):
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = (rows, cols)
        self.nozeros = {}

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, key)
        if type(key) ==  tuple
            if len(key) != 2:
                raise ValueError("Key must have two elements")
            for i in range(2)
                if key in self.nonzeros:
                    return nonzeros[key]
                elif -1 < key[0] && key[0] < self.rows && -1 < key[1] && key[1] < self.cols:
                    return 0
                else:
                    raise IndexError("coordinates out of range")

    def __setitem__(self, key, value):
        if type(key) != tuple:
            raise TypeError("Innapropriate key typpe given -- must be a tuple")
        if len(key) != 2:
            raise ValueError("Key must have two elements")
        if -1 < key[0] && key[0] < self.rows && -1 < key[1] && key[1] < self.cols:
            if value == 0:
                del self.nonzeros[key]
            else:
                self.nonzeros[key] = value
        else:
            raise IndexError("List Index out of range")

    # getslice, setslice deprecated, uses getitem & setitem instead
    # __radd__
    # __rsub__
    # __rmul__
    # __iadd__
    # __isub__
    # __imul__
    # __neg__
    # __invert__
