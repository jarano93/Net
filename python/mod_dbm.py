#!/usr/bin/python

import bsddb
import numpy as np
from numpy.linalg import norm

# SOMETIMES YOU WANT SOMETHING SO LARGE YOUR COMPUTER CAN'T HANDLE IT
# AND SCREW DICTS
# EXTRA GIRTHY 2-D ARRAYS, BREH

class DBMatrix():

    def __init__(self, rows, cols, val, fName):
        self.rows = rows
        self.cols = cols
        self.fName = fName
        self.dbName = fName + '.dbm'
        db = bsddb.hashopen(self.dbName, 'c')
        for r in range(rows):
            for c in range(cols):
                i = c * rows + r
                db['%d'%i] = '%d' % val
        db.sync()
        db.close()

    def get(self, r, c):
        i = c * self.rows + r
        db = bsddb.hashopen(self.dbName, 'r')
        val = float(db[str(i)])
        db.sync()
        db.close()
        return val

    def set(self, r, c, val):
        i = c * self.rows + r
        db = bsddb.hashopen(self.dbName, 'w')
        db['%d'%i] = '%d' % val
        db.sync()
        db.close()

    def sum(self, dbmatrix):
        sum = DBMatrix(self.rows, self.cols, 0, self.fName + '_+_' + dbmatrx.fName)
        for r in range(self.rows):
            for c in range(self.cols):
                sum.set(r,c, self.get(r,c) + dbmatrix.get(r,c))
        return sum

    def transpose(self):
        transpose = DBMatrix(self.cols, self.rows, 0, self.fName + '_T')
        for rt in range(self.cols):
            for ct in range(self.rows):
                transpose.set(rt,ct, self.get(ct, rt))
        return transpose

    def inverse_GS(self):
        # assume the matrix is nonsingular AND square
        gs_TOL = 1e-6
        inverse = DBMatrix(self.rows, self.cols, 0, self.fName + '_inv')
        for c in range(self.cols):
            target = np.zeros((self.rows,1))
            target[c] = 1
            inverse.insert_colvector(self.gauss_seidel_TOL(target, gs_TOL), c)
        return inverse

    def product_col(self, v):
        product = np.zeros((self.rows, 1))
        for r in range(self.rows):
            for c in range(self.cols):
                product[r] += self.get(r,c) * v[r]
        return product

    def transproduct_col(self, v):
        transproduct = np.zeros((self.rows, 1))
        for c in range(self.cols):
            for r in range(self.rows):
                transproduct[c] += self.get(r,c) * v[c]
        return transproduct

    def product_matrix(self, matrix):
        product = DBMatrix(self.rows, matrix.cols, 0, self.fName + '_x_' + matrix.fName)
        for r in range(self.rows):
            for c in range(matrix.cols):
                elem_product = 0
                for rr in range(matrix.rows):
                    for cc in range(self.cols):
                        elem_product += self.get(r, cc) * matrix.get(rr, c)
                product.set(r, c, elem_product)
        return product

    def transproduct_matrix(self, matrix):
        transproduct = DBMatrix(self.cols, matrix.cols, 0, self.fName + '_Tx_' + matrix.fName)
        for rt in range(self.cols):
            for c in range(matrix.cols):
                elem_product = 0
                for rr in range(matrix.rows):
                    for cct in range(self.rows):
                        elem_product += self.get(rt, cct) * matrix.get(rr, c)
                transproduct.set(rt,c, elem_product)
        return transproduct

    def insert_colvector(self, colvector, col):
        for r in range(self.rows):
            self.set(r, col, colvector[r])

    def inf_norm_difference(self, matrix):
        inf_norm = 0
        for r in range(self.rows):
            max_abs = 0
            for c in range(self.cols):
                abs_dif = abs(self.get(r,c) - matrix.get(r,c))
                if abs_dif > max_abs:
                    max_abs = abs_dif
            inf_norm += max_abs
        return inf_norm

    def diag(self):
        diagonal = np.zeros((self.rows, 1))
        for i in range(self.rows):
            diagonal[i] = self.get(i,i)
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

    def gauss_seidel_TOL(self, target, TOL):
        diag = self.diag()
        guess0 = diag * target
        guess1 = diag * target
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
