#!/bin/usr/python

from PIL import Image
import numpy as np

def flat_img(fName):
    return np.asarray(Image.open(fName)).flatten()

def imgArray(npArray, fName):
    Image.fromarray(npArray).save(fName)

def unflat_RGB(flat, vPix, hPix):
    temp = np.array(flat)
    return flat.reshape(vPix, hPix, 4)

def flat_BW(fName):
    return np.asarray(Image.open(fName))[:,:,0].flatten()

def flat_png(fName):
    return np.asarray(Image.open(fName))[:,:,0:3].flatten()

def array_BNW(fName):
    res = np.asarray(Image.open(fName))
    print "%s, shape: %s, reduced to: %s" % (fName, res.shape, res.shape[0:2])
    if len(res.shape) == 2:
        return res
    else:
        return res[:,:,0]

def array_RGB(fName):
    res = np.asarray(Image.open(fName))
    print "%s, shape: %s" % (fName, res.shape)
    return res

def normalize(npArray):
    mean = 0
