from util import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation
import time
import cPickle
import math
import autoencoder
import combine
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score


imgDim = 36
filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
#filters = [grayscale]
(X, Y) = everything(1.0, filters)


#sys.exit(1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.1, random_state=0)


print X[0]
print X[0].shape

print "Training PCA...."
start = time.time()
pca = decomposition.PCA(n_components=16 ** 2, whiten=True).fit(X_train)
print "\t Took: " + str(time.time() - start)
print len(X_train)
print "Transforming data...."
start = time.time()
X_train = pca.transform(X_train)
#X_train = pca.inverse_transform(X_train)
X_test = pca.transform(X_test)
#X_test = pca.inverse_transform(X_test)
print "\t Took: " + str(time.time() - start)

#preview(X_train, 16, 16, int(math.floor(32 * 20 / imgDim)))
#sys.exit(2)


print "\tDeweighting training examples..."
#y_train = np.apply_along_axis(deWeight, 1, y_train)

X_scaler = preprocessing.StandardScaler().fit(np.float32(X_train))
y_scaler = preprocessing.StandardScaler().fit(np.float32(y_train))
X_train_scaled = X_scaler.transform(np.float32(X_train))
y_train_scaled = y_scaler.transform(np.float32(y_train))

def svrFit(X, y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]
    
    clf = GridSearchCV( svm.SVR(kernel='rbf'), tuned_parameters, score_func=r2_score, n_jobs=-1, verbose=0 )
    clf.fit(X, y)
    
    print "Best parameters set found on development set:"
    print clf.best_estimator_

    #clf = svm.SVR(kernel='rbf'), C=1e3, gamma=0.1)
    #clf.fit(np.float32(X), np.float32(y))
    return clf
    
def sgdFit(X, y):
    clf = SGDRegressor(loss="epsilon_insensitive", shuffle=True)    
    clf.fit(np.float32(X), np.float32(y))
    return clf


def scaledPredict(clf, X, i, j):
    p = clf.predict(X_scaler.transform(np.float32(X)))
    return (p + y_scaler.mean_[i:j]) * y_scaler.std_[i:j]


def rfFit(X, y):
    clf = RandomForestRegressor(n_estimators=forestSize, n_jobs=8)
    clf = clf.fit(X, y)
    clf.set_params(n_jobs = 1)
    return clf

def rfPredict(clf, X, i, j): return clf.predict(X)

svmRatio = 0.25
models = [
    { 'fit' : svrFit, 'predict': scaledPredict, 'scaled': True, 'idx0': 0, 'idx1': 1, 'ratio': svmRatio }
    ,{ 'fit' : svrFit, 'predict': scaledPredict, 'scaled': True, 'idx0': 1, 'idx1': 2, 'ratio': svmRatio }
    ,{ 'fit' : svrFit, 'predict': scaledPredict, 'scaled': True, 'idx0': 3, 'idx1': 4, 'ratio': svmRatio }
    ,{ 'fit' : svrFit, 'predict': scaledPredict, 'scaled': True, 'idx0': 4, 'idx1': 5, 'ratio': svmRatio }
    ,{ 'fit' : svrFit, 'predict': scaledPredict, 'scaled': True, 'idx0': 6, 'idx1': 7, 'ratio': svmRatio }
]

#models2 = combine.combineTrain(X_test, y_test, models)

print "Training random forest..."
forestSize = 30
print "\t# Examples: \t\t" + str(len(X_train)) 
print "\tForest Size: \t\t" + str(forestSize)
start = time.time()
clf = RandomForestRegressor(n_estimators=forestSize, n_jobs=8)
clf = clf.fit(X_train, y_train)
print "\tTraining Complete" 
print "\tTime: \t\t" + str(round(time.time() - start, 1)) + "s"

#Reset n_jobs to 1 because multicore evaluation is apparently hard
params = clf.get_params()
clf.set_params(n_jobs = 1)

print "\tRMSE: \t\t" + str(rmse(X_test, y_test, clf.predict, True))
#results = combine.combineTest(X_test, y_test, clf, models)



#def subPredict(X):
#    return combine.combinePredict(X, clf, models)
submission(clf.predict, filters, pca.transform)

