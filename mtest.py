import numpy as np
import random as r
import time
from pymod.meme import MemeNet

data = np.genfromtxt('raw CVR.csv', delimiter=',')

train_size = 35
test_size = 400

trust = 1.5

sets = 1000
reps = 20

total_elems = range(len(data))
train_elems = r.sample(total_elems, train_size)
test_elems = [x for x in total_elems if x not in train_elems]

performance = []

times = []

for s in xrange(sets):
    test_net = MemeNet(len(data[0,1:-1]), 1, 5, 2)
    start = time.time()
    for x in xrange(reps):
        train_err = 0
        single_err = 0
        for i in train_elems:
            single_err = test_net.err_meansq(data[i, 1:-1], data[i,0])
            train_err += single_err 
            # print "single err: %f" % (single_err)
        train_err /= len(train_elems)
        # print "\nTRAIN RUN ERR: %f\n" % (train_err)
        for i in r.sample(train_elems, 1):
            test_net.train(data[i,1:-1], data[i,0], trust, False)
    runtime = time.time() - start
    times.append(runtime)

    test_err = 0
    for i in test_elems:
        single_err = test_net.err_meansq(data[i, 1:-1], data[i,0])
        test_err += single_err
    test_err /= len(test_elems)
    # print "\n******************************"
    print "MEAN RUN ERR: %f" % (test_err)
    # print "******************************\n"
    performance.append(test_err)
print "\nOVERAL PERFORMANCE:"
for i in range(len(performance)):
    print performance[i]
print "\nAVERAGE PERFORMANCE: %f" % (sum(performance) / len(performance))
print "\nSTAND DEVIATION PERFORMANCE: %f" % (np.std(performance))
print "AVERAGE TRAIN TIME: %f" % (sum(times) / len(times))
garbage = raw_input("Press Enter to quit...")
