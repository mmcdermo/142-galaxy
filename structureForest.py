from util import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation
import time
import cPickle

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
(X, Y) = everything(0.5, filters)

#preview(X, 36, 36, 20)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.02, random_state=0)

print "Deweighting training examples..."
y_train = np.apply_along_axis(deWeight, 1, y_train)

print "Training random forest..."
forestSize = 10
print "\t# Examples: \t\t" + str(len(X_train)) 
print "\tForest Size: \t\t" + str(forestSize)
start = time.time()
clf = RandomForestRegressor(n_estimators=forestSize, n_jobs=8)
clf = clf.fit(X_train, y_train)
print "Training Complete" 
print "\tTime: \t\t" + str(round(time.time() - start, 1)) + "s"

#Reset n_jobs to 1 because multicore evaluation is apparently hard
params = clf.get_params()
clf.set_params(n_jobs = 1)

#print "Calculating score..."
#score = clf.score(X_test, y_test)  
#print "\t" + str(score)

def newPredict0(X): return np.array(reWeight(clf.predict(X)[0]))
def newPredict1(X): return reWeight(deWeightedSoftmax(clf.predict(X)[0]))

print "RMSE..."
print "\tReweighted: \t\t" + str(rmse(X_test, y_test, newPredict0))
print "\tSoftmaxed: \t\t" + str(rmse(X_test, y_test, newPredict1))
print "\tUnprocessed:: \t\t" + str(rmse(X_test, y_test, clf.predict))

f = open("randomForest", "wb")
cPickle.dump(clf, f)
f.close()

f = open("randomForest", "r")
clf = cPickle.load(f)
clf.set_params(n_jobs = 1)


#submission(clf.predict, filters)
