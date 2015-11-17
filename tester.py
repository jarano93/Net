import numpy as np
import random as r
from pymod.net import Net

data = np.genfromtxt('krkopt.csv', delimiter=',')
TOL = [100, 10, 1, 1e-1, 1e-2, 1e-3]
train_size = 10000

total_elems = range(len(data))
train_elems = r.sample(total_elems, train_size)
test_elems = [x for x in total_elems if x not in train_elems]

test_net = Net(len(data[0,0:-2], 1)
for i in range(len(TOL)):
    print TOL[i]
    while True:
        train_err = 0
        for i in train_elems:
            train_err += tester.err(data[i,0:-2], data[i,-1])
        print "TOL: %.3f\tTRAIN ERR: %f" % (TOL[i], train_err)
        if train_err < TOL:
            break
        for i in r.sample(train_elems, 100):
            test_net.train_once(data[i,0:-2], data[i,-1])

    test_err = 0
    for i in test_elems:
        test_err += tester.err(data[i,0:-2], data[i,-1])
    print "TOL: %.3f\tTEST ERR: %f" % (TOL[i], train_err)
input("Pres Enter to quit...")
exit()
