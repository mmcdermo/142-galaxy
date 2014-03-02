from util import *

from sklearn import cross_validation
import numpy as np
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer, SigmoidLayer
import cPickle

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]

(X, Y) = everything(0.4, filters)
#preview(X, 36, 36, 20)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.4, random_state=0)

hiddenN = 150
ds = SupervisedDataSet(len(X[0]), len(Y[0]))
for i in range(0, len(X_train)):
    ds.addSample(X_train[i], y_train[i])

net = buildNetwork(len(X[0]), hiddenN, len(Y[0]), bias=True, hiddenclass=SigmoidLayer)
trainer = BackpropTrainer(net, ds)

print "\nTraining Neural Net (" + str(hiddenN) + ") hidden units"
for j in range(0, 5): 
    err = trainer.train()
    print "\tEpoch " + str(j)+ " err: " + str(err)

print "RMSE: " + str(rmse(X_test, y_test, net.activate))

f = open("net", "wb")
cPickle.dump(net, f)
f.close()

f = open("net", "r")
net = cPickle.load(f)
print net


submission(net.activate, filters)


