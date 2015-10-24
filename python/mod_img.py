#!/bin/usr/python

from PIL import Image
import numpy as np

def flat_img(fName):
    return np.asarray(Image.open(fName)).flatten()

def imgArray(npArray, fName):
    temp = np.array(npArray)
    Image.fromarray(temp).save(fName)

def unflat_RGBA(flat, vPix, hPix):
    print type(flat)
    temp = np.array(flat)
    print type(temp)
    return flat.reshape(vPix, hPix, 4)

def map_01_255(vect):
    return np.floor(256 * vect)
