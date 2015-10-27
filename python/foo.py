#!usr/bin/python

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
