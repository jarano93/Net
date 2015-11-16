#!usr/bin/python

import numpy as np
from numpy.linalg import norm

class DictMatrix():
    """
        Store the rows of a Matrix in a dict,
        store rows for faster matrix multiplication
    """

    def __init__(self, rows, cols, default_val=0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = (self.rows, self.cols)
        self.row_dict = {}
        if default_val == 0:
            for r in xrange(self.rows):
                self.row_dict[r] = np.zeros(self.cols)
        elif default_val == 1:
            for r in xrange(self.rows):
                self.row_dict[r] = np.ones(self.cols)
        elif type(default_val) == int or type(default_val) == float:
            for r in xrange(self.rows):
                self.row_dict[r] = default_val * np.ones(self.cols)
        else:
            raise TypeError("Matrix must be initialized with ints or floats")

    def __repr__(self):
        return str(self.rows) + "x" + str(self.cols) + " DictMatrix"

    def __getitem__(self, key):
        """
            Gets either:
            1) Individual value given row & col numbers
            2) Whole col or row given slices & ints
                NO FANCY SPLICES OR SUBMATRICES
        """
        (r,c) = key
        if type(r) == int and type(c) == int:
            if r < -1 or self.rows < r or c < -1 or self.cols < c:
                raise IndexError("Index out of bounds")
            return self.row_dict[r][0,c] # add in 0 so matrix stays cool
        elif type(r) == slice and type(c) == int:
            if c < -1 or self.cols < c:
                raise IndexError("Index out of bounds")
            colvector = np.matrix(np.zeros((self.rows, 1)))
            for row in xrange(self.rows):
                colvector[row] = self.row_dict[row][0,c] # add in 0 so matrix stays cool
            return colvector
        elif type(r) == int and type(c) == slice:
            if r < -1 or self.rows < r:
                raise IndexError("Index out of bounds")
            return self.row_dict[r]

    def __setitem__(self, key, val):
        """
            Sets either:
            1) Individual value given row & col numbers
            2) Whole col or row given slices & ints
                NO FANCY SPLICES OR SUBMATRICES
        """
        (r,c) = key
        if type(r) == int and type(c) == int:
            if r < -1 or self.rows < r or c < -1 or self.cols < c:
                raise IndexError("Index out of bounds")
            self.row_dict[r][0,c] = float(val)
        elif type(r) == slice and type(c) == int:
            if c < -1 or self.cols < c:
                raise IndexError("Index out of bounds")
            if len(val) != self.rows:
                raise ValueError("Cannot set: innapropriate number of elements")
            for row in xrange(self.rows):
                self.row_dict[row][0,c] = val[row] # add in 0 so matrix stays cool
        elif type(r) == int and type(c) == slice:
            if r < -1 or self.rows < r:
                raise IndexError("Index out of bounds")
            if len(val) != self.rows:
                raise ValueError("Cannot set: innapropriate number of elements")
            self.row_dict[r] = np.matrix(val)
 
    @classmethod
    def add(cls, matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Arguments need matching dimensions")
        result = DictMatrix(*matrix1.shape)
        for r in xrange(matrix1.rows):
            result[r,:] = matrix1[r,:] + matrix2[r,:]
        return result

    @classmethod
    def sub(cls, matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Arguments need matching dimensions")
        result = DictMatrix(*matrix1.shape)
        for r in xrange(matrix1.rows):
            result[r,:] = matrix1[r,:] - matrix2[r,:]
        return result

    @classmethod
    def neg(cls, matrix):
        result = DictMatrix(*matrix.shape)
        for r in xrange(matrix.rows):
            result[r,:] = -matrix[r,:]
        return result

    @classmethod
    def mul(cls, matrix, arg):
        if isinstance(arg, DictMatrix):
            return DictMatrix.mul_matrix(matrix, arg)
        if type(arg) == int or type(arg) == float:
            return DictMatrix.mul_scalar(matrix, arg)
        elif type(arg) == np.ndarray:
            return DictMatrix.mul_vector(matrix, arg)

    @classmethod
    def mul_scalar(cls, matrix, scalar):
        result = DictMatrix(*matrix.shape)
        for r in xrange(matrix.rows):
            result[r,:] = scalar * matrix[r,:]
        return result

    @classmethod
    def mul_colvector(cls, matrix, colvector):
        if len(colvector) != matrix.cols:
            raise ValueError("Arguments need matching inner dimensions")
        result = np.zeros((matrix.rows, 1))
        for r in xrange(matrix.rows):
            result[r,:] = matrix[r,:] * colvector
        return result

    @classmethod
    def mul_matrix(cls, matrix1, matrix2):
        if matrix1.cols != matrix2.rows:
            raise ValueError("Arguments need matching inner dimensions")
        result = DictMatrix(matrix1.rows, matrix2.cols)
        for c in xrange(result.cols):
            column = matrix2[:,c]
            for r in xrange(result.rows):
                result[r,c] = matrix1[r,:] * column
        return result

    @classmethod
    def mul_transpose(cls, matrix, transpose):
        if matrix.cols != transpose.cols:
            raise ValueError("Arguments need matching number of columns")
        result = DictMatrix(matrix.rows, transpose.rows)
        for r in xrange(result.rows):
            for c in xrange(result.cols):
                result[r,c] = matrix[r,:] * transpose[c,:].T
        return result

    @classmethod
    def div(cls, matrix, arg):
       if isinstance(arg, DictMatrix):
            return DictMatrix.div_matrix(matrix, arg)
       elif type(arg) == int or type(arg) == float:
            return DictMatrix.div_scalar(matrix, arg)
 
    @classmethod
    def div_scalar(cls, matrix, scalar):
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        result = DictMatrix(*matrix.shape)
        for r in xrange(result.rows):
            result[r,:] = matrix[r,:] / scalar
        return result

    @classmethod
    def div_matrix(cls, matrix1, matrix2):
        """
            Mimics matrix1 * (matrix2 ^ -1) = X by solving the linear system matrix1 = X * matrix2
        """
        if matrix1.cols != matrix2.rows:
            raise ValueError("Arguments do not have matching inner dimensions")
        ls_TOL = 1e-5
        solver = matrix2.get_solver()
        div = DictMatrix(*matrix1.shape)
        for c in xrange(div.cols):
            div[:,c] = solver(matrix1[:,c], ls_TOL)
        return div
 
    def __add__(self, arg):
        return DictMatrix.add(self, arg)

    def __radd__(self, arg):
        return DictMatrix.add(arg, self)

    def __iadd__(self, arg):
        self = DictMatrix.add(self, arg)
        return self

    def __sub__(self, arg):
        return DictMatrix.sub(self, arg)

    def __rsub__(self, arg):
        return DictMatrix.sub(arg, self)

    def __isub__(self, arg):
        self = DictMatrix.sub(self, arg)
        return self

    def __mul__(self, arg):
        return DictMatrix.mul(self, arg)

    def __rmul__(self, arg):
        return DictMatrix.mul(arg, self)
    
    def __imul__(self, arg):
        self = DictMatrix.mul(self, arg)
        return self

    def __div__(self, arg):
        return DictMatrix.div(self, arg)

    def __rdiv__(self, arg):
        if not isinstance(arg, DictMatrix):
            raise TypeError("Leading argument must be a DictMatrix")
        return DictMatrix.div_matrix(arg, self)

    def __idiv__(self, arg):
        self = DictMatrix.div(self, arg)
        return self

    def __mod__(self, arg):
        if not isinstance(arg, DictMatrix):
            raise TypeError("Both arguments must be DictMatrices")
        return DictMatrix.mul_transpose(self, arg)

    def __rmod__(self, arg):
        if not isinstance(arg, DictMatrix):
            raise TypeError("Both arguments must be DictMatrices")
        return DictMatrix.mul_transpose(arg, self)

    def __imod__(self, arg):
        if not isinstance(arg, DictMatrix):
            raise TypeError("Both arguments must be DictMatrices")
        self = DictMatrix.mul_transpose(self, arg)
        return self

    def diag(self):
        if self.rows != self.cols:
            raise ValueError("Matrix not square")
        result = np.zeros((self.rows, 1))
        for i in xrange(self.rows):
            result[i] = self[i,i]
        return np.matrix(result)

    def transpose(self):
        result = DictMatrix(self.cols, self.rows)
        for r in xrange(self.rows):
            for c in xrange(self.cols):
                result[r,c] = self[c,r]
        return result

    def nondiag_jacobi_TOL(self, target, TOL):
        x0 = target
        x1 = x0
        while True:
            for r in xrange(self.rows):
                sum_val = 0
                row = np.array(self[r,:])
                for c in xrange(self.cols):
                    sum_val += row[c] * x0[c]
                x1[r] = x0[r] + target[r] - sum_val
            if norm(x1 - x0, np.inf) < TOL:
                return x1
            x0 = x1

    def gauss_seidel_TOL(self, target, TOL):
        diagonal = self.diag()
        x0 = np.matrix(np.array(diagonal) * np.array(target))
        x1 = x0
        while True:
            for r in xrange(self.rows):
                sum_val = 0
                row = np.array(self[r,:])
                for c in xrange(self.cols):
                    if r > c:
                        sum_val += row[c] * x1[c]
                    elif r < c:
                        sum_val += row[c] * x0[c]
                    x1[r] = (target[r] - sum_val) / diag[r]
            if norm(x1 - x0, np.inf) < TOL:
                return x1
            x0 = x1

    def get_solver(self):
        try:
            diagonal = self.diag()
            if 0 in diagonal:
                return self.nondiag_jacobi_TOL
            else:
                return self.gauss_seidel_TOL
        except ValueError:
            return self.nondiag_jacobi_TOL
