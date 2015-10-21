#!/bin/usr/python

from PIL import Image
import numpy as np

def flat_img(fName):
    return np.asarray(Image.open(fName)).flatten()

def imgArray(npArray, fName):
    Image.fromarray(npArray).save(fName)

def unflat_RGB(flat, vPix, hPix):
    return flat.reshape(vPix, hPix, 3)
