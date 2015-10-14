#!/bin/usr/python

from PIL import Image
import numpy as np

def arrayImg(fName):
    return np.asarray(Image.open(fName))

def imgArray(npArray, fName):
    Image.fromarray(npArray).save(fName)
