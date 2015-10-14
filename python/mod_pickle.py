#!/usr/bin/python

import cPickle

def loadObj(fName):
    f = file(fName, 'rb')
    loadObj = cPicle.load(f)
    f.close()
    return loadObj

def saveObj(obj, fName):
    f = file(fName, 'wb')
    cPickle.dump(obj, f, protocol=cPICKLE.HIGHEST_PROTOCOL)
    f.close()
