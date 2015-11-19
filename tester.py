import numpy as np
import random as r
from pymod.net import Net

data = np.genfromtxt('krkopt.csv', delimiter=',')
TOL = [10, 1, 1e-1, 1e-2, 1e-3]
train_size = 50
test_size = 1000

total_elems = range(len(data))
train_elems = r.sample(total_elems, train_size)
test_elems = [x for x in total_elems if x not in train_elems]

test_net = Net(len(data[0,0:-2]), 1, 4e3, 1e3, 1e3, False)
single_err = 0
for t in range(len(TOL)):
    for n in xrange(40):
        train_err = 0
        for i in train_elems:
            single_err = test_net.err(data[i,0:-2], data[i,-1])
            print "TOL: %.3f\tSINGLE RUN ERR: %f" % (TOL[t], single_err)
            train_err += single_err
        train_err /= len(train_elems)
        print "\nTOL: %.3f\tMEAN RUN ERR: %f\n" % (TOL[t], train_err)
        if train_err < TOL[t]:
            break
        for i in r.sample(train_elems, 15):
            test_net.train_once(data[i,0:-2], data[i,-1], True, step_size=7e-4)

    test_err = 0
    for i in r.sample(test_elems, 1000):
        single_err = test_net.err(data[i,0:-2], data[i,-1])
        print "TOL: %.3f\tSINGLE RUN ERR: %f" % (TOL[t], single_err)
        test_err += single_err
    test_err /= test_size
    print "\nTOL: %.3f\tMEAN RUN ERR: %f\n" % (TOL[t], test_err)
    # raw_input("Press Enter to continue")
raw_input("Pres Enter to quit...")
exit()