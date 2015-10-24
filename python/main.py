#!/usr/bin/python

import mod_net as NET
import mod_img as IMG
import mod_pickle as PKL
import numpy as np
# data created using http://gmaps-samples-v3.googlecode.com/svn/trunk/styledmaps/wizard/index.html

target = IMG.flat_img('../target.png')

input = IMG.flat_img('../elevmap.png')
input = np.append(input, IMG.flat_img('../roadmap.png')
input = np.append(input, IMG.flat_img('../landmap.png')

net_GD = NET.Net(len(input), len(target), 1e6, 2e6, 5e5)
result_GD_100 = net_GD.train_N_GD(input, target, 100, 1e-6)
IMG.imgArray(IMG.unflat_RGBA(result_GD_100, '../resGD100.png'))

net_CGD = NET.Net(len(input), len(target), 1e6, 2e6, 5e5)
result_CGD = net_CGD.train_N_CGD(input, target, 100)
IMG.imgArray(IMG.unflat_RGBA(result_CGD_100, '../resCGD100.png'))
