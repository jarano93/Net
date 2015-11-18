import numpy as np
from pymod.net import Net
import random as r
import time

data = np.genfromtxt('raw CVR.csv', delimiter=',')
reps = 100

performance = []
train_size = 35
test_size = 400

total_elems = range(len(data))
train_elems = r.sample(total_elems, train_size)
test_elems = [x for x in total_elems if x not in train_elems]

single_err = 0

start = time.time()
for rep in range(reps):
    test_net = Net(len(data[0,0:-2]), 1, 3, 3, 3)
    for n in xrange(1000):
        train_err = 0
        accuracy = 0
        for i in train_elems:
            single_err = test_net.err(data[i,0:-2], data[i,-1])
            output = test_net.feedforward(data[i,0:-2])[0]
            if abs(output - data[i,0]) < 1:
                accuracy += 1
            print "SINGLE RUN ERR: %f" % (single_err)
            train_err += single_err
        train_err /= len(train_elems)
        accuracy = float(accuracy) / train_size
        print "MEAN RUN ERR: %f\n" % (train_err)
        print "MEAN RUN ACC: %f\n" % (accuracy)
        raw_input("Enter...")
        if accuracy > 0.85 and train_err < 5e-4:
            break
        for i in r.sample(train_elems, 35):
            test_net.train_once(data[i,1:-1], data[i,0], True, learn=0.1, forget=1e-3)
            # print "output: %s\ttarget: %s" % (str(test_net.feedforward(data[i,1:-1])[0]), str(data[i,0]))

    test_err = 0
    accuracy = 0
    output = 0
    for i in r.sample(test_elems, test_size):
        single_err = test_net.err(data[i,0:-2], data[i,-1])
        print "SINGLE TEST ERR: %f" % (single_err)
        output = test_net.feedforward(data[i,0:-2])[0]
        if abs(output - data[i,-1]) < 1:
            accuracy += 1
            print accuracy
        print "output: %s\ttarget: %s" % (str(output), str(data[i,0]))
        test_err += single_err
    test_err /= test_size
    accuracy = float(accuracy) / test_size
    performance.append(accuracy)
    print "\nN: %i" % (rep)
    print "MEAN TEST ERR: %f" % (test_err)
    print "PERCENT ACCURACY: %f" % (accuracy)
    print "TRAINS: %i" % (n)
    print "RUNNING ACCURACY: %f" % (float(sum(performance)) / len(performance))
    print "RUNNING TIME: %f" % ((time.time() - start) / len(performance))
    # raw_input("Press Enter to continue")
print "\n\nTOTAL MEAN ACCURACY: %f" % (float(sum(performance)) / len(performance))
print "AVERAGE TIME: %f" % ((time.time() - start) / len(performance))
raw_input("Pres Enter to quit...")
exit()
