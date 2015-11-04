#!/usr/bin/python

import bsddb
import numpy as np
from numpy.linalg import norm

# SOMETIMES YOU WANT SOMETHING SO LARGE YOUR COMPUTER CAN'T HANDLE IT
# AND SCREW DICTS
# EXTRA GIRTHY 2-D ARRAYS, BREH
# NONSPARSE BRUH, NONSPARSE ;_;
# Reverse Polish Notation now too.

class DBMatrix():

    def __init__(self, rows, cols, val, fName):
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = (rows, cols)
        self.fName = fName
        self.dbName = fName + '.dbm'
        db = bsddb.hashopen(self.dbName, 'c')
        for r in range(self.rows):
            for c in range(self.cols):
                i = c * rows + r
                db['%d'%i] = '%d' % val
        db.sync()
        db.close()

    def __getitem__(self, key):
        """
            Gets either:
            1) Individual value given row & col numbers
            2) Whole col or row given splices & ints
                NO FANCY SPLICES OR SUBMATRICES
        """
        (r,c) = key
        db = bsddb.hashopen(self.dbName, 'r')
        if type(r) == int && type(c) == int:
            if r < -1 || self.rows < r || c < -1 || self.cols < c:
                db.sync()
                db.close()
                raise IndexError("index out of bounds")
            i = c * self.rows + r
            val = float(db[str(i)])
            return val
        elif type(r) == splice && type(c) == int: # get a whole column
            if c < -1 || self.cols - 1  < c:
                db.sync()
                db.close()
                raise IndexError("column index out of bounds")
            col_vector = np.zeros((self.rows, 1))
            col = c * self.rows
            for i in xrange(self.rows):
                row_index = col + i
                col_vector[i] = float(db[str(row_index)])
            return col_vector
        elif type(r) == int && type(c) == splice:
            if r < -1 || self.rows - 1 < r:
                db.sync()
                db.close()
                raise IndexError("row index out of bounds")
            row_vector = np.zeros((1, self.cols))
            for j in xrange(self.cols):
                col_index = (j * self.rows) + r
                row_vector[1, j] = float(db[str(col_index)]_
            return row_vector
        db.sync()
        db.close()

    def __setitem__(self, key, val):
        """
            Sets either:
            1) Individual value given row & col numbers
            2) Whole col or row given splices & ints
                NO FANCY SPLICES OR SUBMATRICES
        """
        (r,c) = key
        db = bsddb,hasopen(self.dbName, 'w')
        if type(r) == int && type(c) == int:
            if r < -1 || self.rows < r || c < -1 || self.cols < c:
                db.sync()
                db.close()
                raise IndexError("index out of bounds")
            i = c * matrix.rows + r
            db['%d'%i] = '%d' % val
        elif type(r) == splice && type(c) == int:
            if c < -1 || self.cols - 1  < c:
                db.sync()
                db.close()
                raise IndexError("column index out of bounds")
            if len(val) != self.rows:
                db.sync()
                db.close()
                raise ValueError("New number of rows does not match original")
            col = c * self.rows
            for i in xrange(self.rows):
                row_index = col + i
                db['%d'%row_index] = '%d' % val[i]
        elif type(r) == int && type(c) == splice:
            if r < -1 || self.rows - 1 < r:
                db.sync()
                db.close()
                raise IndexError("row index out of bounds")
            if len(val) != self.cols:
                db.sync()
                db.close()
                raise ValueError("New number of columns does not match original")
            for j in xrange(self.cols):
                col_index = (j * self.rows) + r
                db['%d'%col_index] = '%d' % val[j]
        db.sync()
        db.close()

    @classmethod
    def add(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Argument matrices do not have matching shape")
        add_name = matrix1.fName + '_' + matrix2.fName + '_add'
        add = DBMatrix(matrix1.rows, matrix1.cols, 0, add_name)
        for r in xrange(add.rows):
            for c in xrange(add.cols):
                add[r,c] = matrix1[r,c] + matrix2[r,c]
        return add

    @classmethod
    def neg(matrix):
        neg_name = matrix.fName + '_neg'
        neg = DBMatrix(matrix.rows, matrix.cols, 0, neg_name)
        for r in xrange(self.rows):
            for c in xrange(self.cols):
                neg[r,c] = -1 * matrix[r,c]
        return neg

    @classmethod
    def sub(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Argument matrices do not have matching shape")
        sub_name = matrix1.fName + '_' + matrix2.fName + '_sub'
        sub = DBMatrix(matrix1.rows, matrix1.cols, 0, sub_name)
        for r in xrange(sub.rows):
            for c in xrange(sub.cols):
                sub[r,c] = matrix1[r,c] - matrix2[r,c]
        return sub

    @classmethod
    def mul_scalar(matrix, scalar):
        mul_name = matrix.fName + '_' str(scalar) + '_mul'
        mul = DBMatrix(matrix.rows, matrix.cols, 0, mul_name)
        for r in xrange(matrix.rows):
            for c in xrange(matrix.cols):
                mul[r,c] = scalar * matrix[r,c]
        return mul

    @classmethod
    def mul_vector(matrix, vector):
        if matrix.rows != len(vector):
            raise ValueError("Arguments do not have matching inner dimensions")
        product = np.zeros((matrix.rows, 1))
        for r in xrange(matrix.rows):
            product[r] = matrix[r,:] * vector
        return product

    @classmethod
    def mul_matrix(matrix1, matrix2):
        if matrix1.cols != matrix2.rows:
            raise ValueError("Argument matrices do not have matching inner dimensions")
        mul_name = matrix1.fName + '_' + matrix2.fName + '_mul'
        mul = DBMatrix(matrix1.rows, matrix2.cols, 0, mul_name)
        for r in xrange(matrix1.rows):
            for c in xrange(matrx2.cols):
                mul[r,c] = matrix1[r,:] * matrix2[:,c]
        return mul

    @classmethod
    def mul_transpose(matrix1, matrix2):
        """ Mimics A^T * B """
        if matrix1.rows != matrix2.rows:
            raise ValueError("Argument matrices do not have matching inner dimensions")
        mul_name = matrix1.fName + '_T_' + matrix2.fName + '_mul'
        mul = DBMatrix(matrix1.cols, matrix2.cols, 0, mul_name):
        for rt in xrange(matrix1.cols):
            for c in xrange(matrix2.cols):
                mul[rt,c] = matrix1[:,rt].T * matrix2[:,c]
        return mul

    @classmethod
    def div_scalar(matrix, scalar):
        if scalar == 0:
            raise ValueError("Can't divide by zero!")
        div_name = matrix.fName + '_' + str(scalar) + '_div'
        div = DBMatrix(matrix.rows, matrix.cols, 0, div_name
        for r in xrange(matrix.rows):
            for c in xrange(matrix.cols):
                div[r,c] = matrix[r,c] / scalar
        return div

    @classmethod
    def div_matrix(matrix1, matrix2):
        """
            Mimics matrix1 * (matrix2 ^ -1) = X by solving the linear system matrix1 = X * matrix2
        """
        if matrix1.cols != matrix2.rows:
            raise ValueError("Arguments do not have matching inner dimensions")
        ls_TOL = 1e-5
        solver = matrix2.linsys_solv()
        div_name = matrix1.fName + '_' + matrix2.fName + '_div'
        div = DBMatrix(matrix1.rows, matrix1.cols, 0, div_name)
        for c in range(matrix1.cols):
            div[:,c] = solver(matrix1[:,c], ls_TOL)
        return div

    def __add__(self, other):
        return DBMatrix.add(self, other)

    def __radd__(self, other):
        return DBMatrix.add(self, other)

    def __iadd__(self, other):
        return DBMatrix.add(self, other)

    def __sub__(self, other):
        return DBMatrix.sub(self, other)

    def __rsub__(self, other):
        return DBMatrix.sub(other, self)

    def __isub__(self, other):
        return DBMatrix.sub(self, other)

    def __mul__(self, other):
        if type(other) == int || type(other) == float;
            return DBMatrix.mul_scalar(self, other)
        elif type(other) = np.ndarray:
            return DBMatrix.mul_vector(self, other)
        elif type(other) == DBMatrix:
            return DBMatrix.mul_matrix(self, other)

    def __rmul__(self, other):
        return other.__mul__(self)

    def __imul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if type(other) == int || type(other) == float:
            return DBMatrix.div_scalar(matrix, other)
        elif type(other) == DBMatrix:
            return DBMatrix.div_matrix(matrix, other)

    def __idiv__(self, other):
        return self.__div__(other)

    def transpose(self):
        transpose = DBMatrix(self.cols, self.rows, 0, self.fName + '_T')
        for rt in range(self.cols):
            for ct in range(self.rows):
                transpose.set(rt,ct, self.get(ct, rt))
        return transpose

    def diag(self):
        diagonal = np.zeros((self.rows, 1))
        for i in range(self.rows):
            diagonal[i] = self[i,i]
        return diagonal

    def jacobi_TOL(self, target, TOL):
        diag = self.diag()
        x0 = diag * target
        x1 = x0
        while True:
            for r in range(self.rows):
                sum = 0
                for c in range(self.cols):
                    if r != c:
                        sum += self.get(r,c) * x0[c]
                x1[r] = (x1[r] - sum) / diag[r]
            if norm(x1 - x0, np.inf) < TOL:
                return x1
            x0 = x1

    def nondiag_jacobi_TOL(self, target, TOl):
        diag = self.diag()
        x0 = diag * target
        x1 = x0
        while True:
            for r in range(self.rows):
                sum = 0
                for c in range(self.cols):
                    sum += self.get(r,c) * x0[c]
                x1[r] = x0[r] + target[r] - sum
        if norm(x1 - x0, np.inf) < TOL:
            return x1
        x0 = x1

    def gauss_seidel_TOL(self, target, TOL):
        diag = self.diag()
        guess0 = diag * target
        guess1 = guess0
        while True:
            for r in range(self.rows):
                sum = 0
                for c in range(self.cols):
                    if r > c:
                        sum += self.get(r,c) * guess1[c]
                    elif r < c:
                        sum += self.get(r,c) * guess0[c]
                    guess1[r] = (target[r] - sum) / diag[r]
            if norm(guess1 - guess0, np.inf) < TOL:
                return guess1
            guess0 = guess1

    def linsys_solv(self):
        if self.rows == self.cols:
            diag = self.diag()
            if 0 in diag:
                return self.nondiag_jacobi_TOL
            else:
                return self.gauss_seidel_TOL
        else:
            return self.nondiag_jacobi_TOL
