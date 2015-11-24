import numpy as np
import random as r
import time
from pymod.meme import MemeFFNN
import pymod.meme as meme
import pymod.prep as prep

data = np.genfromtxt('raw CVR.csv', delimiter=',')

num_principals = 4

train_size = 35
test_size = 400

trust = 1e-1

sets = 100

train_slice = slice(1,None, None)
target_slice = slice(0,1)

observations = data[:,train_slice]
observations = prep.pca(observations, num_principals)

targets = data[:,target_slice]

performance = []
accuracies = []
times = []

for s in xrange(sets):
        test_net = MemeFFNN(num_principals, 1, 12, 7)
        total_elems = range(len(data))
        train_elems = r.sample(total_elems, train_size)
        train_set = np.array([observations[train_elems[0],:]])
        target_set = [targets[train_elems[0]]]
        for i in train_elems[1:]:
            train_set = np.append(train_set, [observations[i,:]], axis=0)
            target_set = np.append(target_set, [targets[i,target_slice]], axis=0)
        test_elems = [x for x in total_elems if x not in train_elems]

        print "initial set mean sq err: %f" % (test_net.set_meansq(train_set, target_set))
        start = time.time()
        test_net.train_set_TOL(train_set, target_set, trust, True)
        train_time = time.time() - start
        times.append(train_time)
        
        test_err = 0
        test_acc = 0
        for i in test_elems:
            single_err = test_net.err_meansq(observations[i,:], targets[i])
            test_err += single_err
            if abs(test_net.forgetforward(observations[i,:])[0] - targets[i]) < 1:
                test_acc += 1
        test_err /= len(test_elems)
        test_acc = float(test_acc) / len(test_elems) * 100
        print "\n******************************"
        print "TRAIN TIME: %fs" % (train_time)
        print "MEAN RUN ERR: %f" % (test_err)
        print "MEAN RUN ACC: %f%%" % (test_acc)
        print "******************************\n"
        # raw_input("Press 'Enter' to continue...")
        performance.append(test_err)
        accuracies.append(test_acc)
print "\n****OVERAL PERFORMANCE***"
print "AVERAGE TRAIN TIME: %f" % (sum(times) / len(times))
print "AVERAGE MEAN SQ ERR: %f" % (sum(performance) / len(performance))
print "AVERAGE ACCURACY: %f%%" % (sum(accuracies) / len(accuracies))
print "MIN ACCURACY: %f%%" % (min(accuracies) * 100)
print "MAX ACCURACY: %f%%" % (max(accuracies) * 100)
print "STD TIMES: %f" % (np.std(times))
print "STD MEAN SQ ERR: %f" % (np.std(performance))
print "STD ACCURACY: %f%%" % (np.std(accuracies))
garbage = raw_input("Press 'Enter' to quit...")
