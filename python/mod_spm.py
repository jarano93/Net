#!/usr/bin/python

class Sparse():

    def __init__(self, rows, cols):
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = (rows, cols)
        self.nozero = {}
        self.threshold = 1e-3
        
    def transpose(self):
        transpose = Sparse(self.cols, self.rows):
        for coord in self.nonzero():
            coordT = (coord[1], coord[0])
            transpose.nonzero[coordT] = self.nonzero[coord]
        return transpose

    @classmethod
    def add(matrix1, matrix2):
        """ matrix1 + matrix2 """
        if matrix1.shape != matrix2.shape:
            raise ValueError("Dimensions do not match")
        sum = Sparse(matrix1.rows, matrix2.cols)
        sum.nonzero = matrix1.nonzero
        for key in matrix2.nonzero:
            if key not in matrix1.nonzero:
                sum.nonzero[key] = matrix2.nonzero[key]
            else:
                val = matrix1.nonzero[key] + matrix2.nonzero[key]
                if abs(val) > sum.threshold:
                    sum.nonzero[key] = val
                else:
                    del sum.nonzero[key]
        return sum

    @classmethod
    def sub(matrix1, matrix2):
        """ matrix1 - matrix2 """
        if matrix1.shape != matrix2.shape:
            raise ValueError("Dimensions do not match")
        dif = Sparse(matrix1.rows, matrix1.cols)
        dif.nonzero = matrix1.nonzero:
        for key in matrix2.nonzero:
            if key not in matrix1.nonzero:
                dif.nonzero[key]  = - matrix2.nonzero[key]
            else:
                val = matrix1.nonzero[key] - matrix2.nonzero[key]
                if abs(val) > sum.threshold:
                    sum.nonzero[key] = val
        return dif

    @classmethod
    def scalar_product(matrix, scalar):
        product = Sparse(matrix.rows, matrix.cols)
        for r in product.rows:
            for c in product.rows:
                val = matrix[r,c] * scalar
                if abs(val) > product.threshold:
                    product[r,c] = val
        return product

    @classmethod
    def vector_product(matrix, vector):
        if len(vector) != matrix.cols:
            raise ValueError("Inner dimensions do not match")
        product = Sparse(1, matrix.cols)
        for r in matrix.rows:
            sum = 0
            for c in matrix.cols:
                sum += matrix[r,c] * vector[c]
            if abs(sum) > product.threshold:
                product[r,1] = sum
        return product

    @classmethod
    def matrix_product(matrix1, matrix2):
        if matrix1.cols != matrix2.rows:
            raise ValueError("Inner dimensions do not match")
        product = Sparse(matrix1.rows, matrix2.cols)
        for r in matrix1.rows

    @classmethod
    def handle_mul(matrix, arg, reverse=False):
        if arg.shape != None:
            if reverse:
                return Sparse.matrix_product(arg, matrix)
            else:
                return Sparse.matrix_product(matrix, arg)
        try:
            len(arg)
            return Sparse.vector_proudct(matrix, arg)
        except TypeError as e:
            return Sparse.scalar_product(matrix,arg)

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, key)
        if type(key) ==  tuple
            if len(key) != 2:
                raise ValueError("Key must have two elements")
            (r,c) = key
            if type(r) == int && type(c) == int:
                if -1 < r && r < self.rows && -1 < c && c < self.cols:
                    return self.nonzero[key]
                else:
                    raise IndexError("Coordinates out of bounds")
            elif type(r) == slice && type(c) == int:
                if -1 < c && c < self.cols:
                    if r.start != None && r.stop != None:
                        if r.start < -1 || self.rows < r.stop:
                            raise IndexError("Slice exceeds shape")
                        subrows = range(r.start, r.stop, r.step)
                    else:
                        subrows = len(self.rows)
                    submatrix = Sparse(len(subrows), 1)
                    subindex = 0
                    for i in subrows:
                        if (i,c) in self.nonzero:
                            submatrix.nonzero[(subindex, 1)] = self.nonzero[(i,c)]
                        subindex++
                    return submatrix
                else:
                    raise IndexError("Coordinates out of bounds")
            elif type(r) == int && type(c) == slice:
                if -1 < r && r < self.rows:
                    if c.start != None && c.stop != None:
                        if c.start < -1 || self.rows < c.stop:
                            raise IndexError("Slice exceeds shape")
                        subcols = range(c.start, c.stop, c.step)
                    else:
                        subcols = len(self.cols)
                    submatrix = Sparse(1, len(subcols))
                    subindex = 0
                    for j in subcols:
                        if (r,j) in self.nonzero:
                            submatrix.nonzero[(1 ,subindex)] = self.nonzero[(r,j)]
                        subindex++
                    return submatrix
                else:
                    raise IndexError("Coordinates out of bounds")
            else: # both slices
                if r.start != None && r.stop != None:
                    if r.start < -1 || self.rows < r.stop:
                        raise IndexError("Slice exceeds shape")
                    subrows = range(r.start, r.stop, r.step)
                else:
                    subrows = len(self.rows)
                if c.start != None && c.stop != None:
                    if c.start < -1 || self.rows < c.stop:
                        raise IndexError("Slice exceeds shape")
                    subcols = range(c.start, c.stop, c.step)
                else:
                    subcols = len(self.cols)
                submatrix = Sparse(len(subrows), len(subcols))
                subr = 0
                subc = 0
                for i in subrows:
                    for j in subcols:
                        if (i,j) in self.nonzero:
                            submatrix.nonzero[(subr, subc)] = self.nonzero[(i,j)]
                        subr++
                        subc++
                return submatrix
        else:
            raise IndexError("Coordinates must be in tuple form")

    def  __setitem__(self, key, set):
        if type(key) ==  tuple
            if len(key) != 2:
                raise ValueError("Key must have two elements")
            (r,c) = key
            if type(r) == int && type(c) == int:
                if -1 < r && r < self.rows && -1 < c && c < self.cols:
                    if abs(set) > self.threshold:
                        self.nonzero[key] = set
                else:
                    raise IndexError("Coordinates out of bounds")
            elif type(r) == slice && type(c) == int:
                if -1 < c && c < self.cols:
                    if r.start != None && r.
                        if r.start < -1 || self.rows < r.stop:
                            raise IndexError("Slice exceeds shape")
                        subrows = range(r.start, r.stop, r.step)
                        if len(subrows) != len(set):
                            raise ValueError("New value not same shape as original value")
                    else:
                        subrows = len(self.rows)
                    setindex = 0
                    for i in subrows:
                        val = set[setindex]
                        if abs(val) > self.threshold:
                            self.nonzero[(i, c)] = val
                        setindex++
                else:
                    raise IndexError("Coordinates out of bounds")
            elif type(r) == int && type(c) == slice:
                if -1 < r && r < self.rows:
                    if c.start < -1 && self.cols < c.stop:
                        raise IndexError("Slice exceeds shape")
                    subcols = range(c.start, c.stop, c.step)
                    if len(subcols) != len(set):
                        raise ValueError("New value not same shape as orignal value")
                    setindex = 0
                    for j in subcols:
                        val = set[subindex]
                        if abs(val) > self.threshold: 
                            self.nonzero[(r ,j)] = val
                        subindex++
                    return submatrix
                else:
                    raise IndexError("Coordinates out of bounds")
            else: # both slices
                if -1 < r.start && r.stop < self.rows && -1 < c.start && c.stop < self.cols:
                    subrows = range(r.start, r.stop, r.step)
                    subcols = range(c.start, c.stop, c.step)
                    if len(subrows) != len(set) && len(subcols) != len(set[0]):
                        raise ValueError("New value not same shape as original value")
                    subr = 0
                    subc = 0
                    for i in subrows:
                        for j in subcols:
                            val = set[subr, subc]
                            if abs(val) > self.threshold:
                                self.nonzero[(i, j)] = val
                            subr++
                            subc++
                else:
                    raise IndexError("Coordinates out of bounds")
        else:
            raise IndexError("Coordinates must be in tuple form")

    # getslice, setslice deprecated, uses getitem & setitem instead
    def  __add__(self, matrix):
        return Sparse.add(self, matrix)

    def __radd__(self, matrix):
        return Sparse.add(self, matrix)
    
    def __iadd__(self, matrix):
        return Sparse.add(self, matrix)

    def __sub__(self, matrix):
        return Sparse.sub(self, matrix)

    def __rsub__(self, matrix):
        return Sparse.sub(matrix, self)

    def __isub__(self, matrix):
        return Sparse.sub(self, matrix)

    def __mul__(self, arg):
        return Sparse.handle_mul(self,arg)
    
    def __rmul__(self, arg):
        return Sparse.handle_mul(
       
    # __imul__
    # __neg__
    # __invert__
