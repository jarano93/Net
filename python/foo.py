#!usr/bin/python

from mod_dbm import DBMatrix
import time

def time_dims(rows, cols):
    start = int(time.time())
    DBMatrix(rows, cols, 0, 'test')
    return int(time.time()) - start
    
class Ayy():
    
    def __getitem__(self, key):
        print key
        print type(key)
        for i in range(len(key)):
            print key[i]
            print type(key[i])

    def __setitem__(self, key, val):
        print key
        print val
        print type(val)
    def __getslice__(self, i, j):
        print i
        print j
    def __setslice__(self, i, j, v):
        print i
        print j
        print v
