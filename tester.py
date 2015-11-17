import numpy as np
import random as r
from pymod.net import Net

data = np.genfromtxt('krkopt.csv', delimiter=',')

train_size = 10000
total_elems = range(len(data))
train_elems = r.sample(total_elems, train_size)
train_elems = [x for x in total_elems if x not in train_elems]

input("Pres Enter to quit...")
exit()
