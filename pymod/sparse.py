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
    def mul_scalar(matrix, scalar):
        product = Sparse(matrix.rows, matrix.cols)
        for r in product.rows:
            for c in product.rows:
                val = matrix[r,c] * scalar
                if abs(val) > product.threshold:
                    product[r,c] = val
        return product

    @classmethod
    def mul_vector(matrix, vector):
        if len(vector) != matrix.cols:
            raise ValueError("Inner dimensions do not match")
        product = Sparse(1, matrix.cols)
        for key in matrix.nonzero:
            (r,c) = key
            val = matrix[key] * vector[c] 
            if abs(val) < product.threshold:
                continue
            if c not in product.nonzero:
                product[c] = val
            else:
                product[c] += val
        for key in product.nonzero:
            if abs(product[key]) < product.threshold:
                del product.nonzero[key]
        return product

    @classmethod
    def mul_matrix(matrix1, matrix2):
        if matrix1.cols != matrix2.rows:
            raise ValueError("Inner dimensions do not match")
        product = Sparse(matrix1.rows, matrix2.cols)
        for key1 in matrix1.nonzero:
            for key2 in matrix2.nonzero:
                (r1,c1) = key1
                (r2,c2) = key2
                if c1 == r2:
                    val = matrix1[r1,c1] * matrix2[r2,c2]
                    if abs(val) < product.threshold:
                        continue
                    if (r1,c2) not in product.nonzero:
                        product[r1,c2] = val
                    else:
                        product[r1,c2] += val
        for key in product.nonzero:
            if abs(product[key]) < product.threshold:
                del product.nonzero[key]
        return product

    @classmethod
    def mul_transpose(matrix1, matrix2):

    @classmethod
    def div_scalar(matrix, scalar):
        if scalar == 0:
            raise ValueError("Can't divide by zero!")
        div = Sparse(*matrix.shape)
        for key in div.nonzero:
            val = matrix[key] / scalar
            if abs(val) > div.threshold:
                div.nonzero[key] = val
        return div

    @classmethod
    def div_matrix(matrix1, matrix2):
        """
            Mimics matrix1 * (matrix2 ^ -1) = X by solving the linear system matrix1 = X * matrix2
        """
        if matrix1.cols != matrix2.rows:
            raise ValueError("Arguments do not have matching inner dimensions")
        ls_TOL = 1e-5
        div = Sparse(*matrix1.shape)
        

    def __getitem__(self, key):
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
        try:
            arg_shape = arg.shape
            if len(arg_shape) == 1:
                return Sparse.mul_vector(self, arg)
            elif len(arg_shape) == 2:
                (r,c) = arg_shape
                if c == 1:
                    return Sparse.mul_vector(self, arg)
                else:
                    return Sparse.mul_matrix(self, arg)
            else:
                raise ValueError("Arguments have too high dimension")
        except AttributeError:
            if type(other) == int or type(other) == float:
                return Sparse.mul_scalar(self, other)
            else:
                raise TypeError("Argument of wrong type")
    
    def __rmul__(self, arg):
        return arg.__mul__(self)
       
    def __imul__(self, arg):
        return self.__mul__(arg)

    # __neg__
    # __div__
    # __mod__
