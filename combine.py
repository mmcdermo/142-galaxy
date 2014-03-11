from util import *
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
import time
import math

def combinePredict(X, base, models):
    Xp = base.predict(X)
    for model in models:
        i = model['idx0']
        j = model['idx1']
        Xp[i:j] = model['predict'](model['trained'], X, i, j)
    return Xp

def combineTest(X_test, y_test, base, models):
    for model in models:
        i = model['idx0']
        j = model['idx1']

        print "RMSE for model for indices " + str(i) + ":" + str(j)        
        z = lambda(X): model['predict'](model['trained'], X, i, j)
        print "\t" + str(rmse(X_test, y_test[0:len(y_test), i:j], z, True))
        
    print "Combined RMSE:"

    def combineRaw(X): return combinePredict(X, base, models)
    print "\tCombined Raw: \t\t" + str(rmse(X_test, y_test, combineRaw, True))
'''        
    def combineReweight(X): 
        return np.array(reWeight(combinePredict(X, base, models)[0]))
    print "\tCombined Reweighted: \t\t" + str(rmse(X_test, y_test, combineReweight))

    def combineSoftmaxed(X): 
        return np.array(deWeightedSoftmax(reWeight(combinePredict(X, base, models)[0])))
    print "\tCombined Softmaxed: \t\t" + str(rmse(X_test, y_test, combineSoftmaxed))
'''
def combineTrain(X_train, y_train, models):    
    X_scaler = preprocessing.StandardScaler().fit(np.float32(X_train))
    y_scaler = preprocessing.StandardScaler().fit(np.float32(y_train))
    X_train_scaled = X_scaler.transform(np.float32(X_train))
    y_train_scaled = y_scaler.transform(np.float32(y_train))
    for m in range(len(models)):
        X = X_train
        y = y_train
        model = models[m]
        if model['scaled'] == True:
            X = X_train_scaled
            y = y_train_scaled
        i = model['idx0']
        j = model['idx1']
        lim = math.floor(model['ratio'] * len(y))
        print "Training model for indices " + str(model['idx0']) + ":" + str(model['idx1'])
        start = time.time()
        y_ij = y[0:lim, i:j].ravel()
        models[m]['trained'] = model['fit'](X[0:lim], y_ij)
        print "\tTook " + str(time.time() - start) + "s"
    return models
