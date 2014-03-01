from util import *
from math import sqrt
from sklearn import cross_validation
import numpy as np
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer, SigmoidLayer

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
(X, Y) = loadProcess(0.25, filters)

preview(X, 36, 36, 20)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.4, random_state=0)

hiddenN = 25
ds = SupervisedDataSet(len(X[0]), len(Y[0]))
for i in range(0, len(X_train)):
    ds.addSample(X_train[i], y_train[i])

net = buildNetwork(len(X[0]), hiddenN, len(Y[0]), bias=True, hiddenclass=SigmoidLayer)
trainer = BackpropTrainer(net, ds)

print "\nTraining Neural Net (" + str(hiddenN) + ") hidden units"
for j in range(0, 3): 
    err = trainer.train()
    print "\tEpoch " + str(j)+ " err: " + str(err)

#Get real error
errs = []
for i in range(0, len(X_test)):
    activ = net.activate(X_test[i])
    diff = y_test[i] - activ
    diffSq = np.vectorize(lambda(x): x ** 2)(diff)
    errs.append(np.sum(diffSq))

errSum = np.sum(np.array(errs))
rmse = sqrt( errSum / (len(X_test) * len(X_test[0])) )
print "RMSE: " + str(rmse)


