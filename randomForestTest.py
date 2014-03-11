from util import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation
import cPickle

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
(X, Y) = everything(0.01, filters)

#preview(X, 36, 36, 20)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.4, random_state=0)

print "Training random forest..."
clf = RandomForestRegressor(n_estimators=10, n_jobs=8)
clf = clf.fit(X_train, y_train)


#Reset n_jobs to 1 because multicore evaluation is apparently hard
params = clf.get_params()
clf.set_params(n_jobs = 1)

print "Calculating score..."
score = clf.score(X_test, y_test)  
print "\t" + str(score)
print "RMSE..."
print "\t" + str(rmse(X_test, y_test, clf.predict))

f = open("randomForest", "wb")
cPickle.dump(clf, f)
f.close()

f = open("randomForest", "r")
clf = cPickle.load(f)


print rmse(X_test, y_test, clf.predict, True)
submission(clf.predict, filters)
