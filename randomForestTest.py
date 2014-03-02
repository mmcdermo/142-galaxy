from util import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
(X, Y) = everything(1.0, filters)

preview(X, 36, 36, 20)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.4, random_state=0)

clf = RandomForestRegressor(n_estimators=10, n_jobs=8)
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)  
print score

params = clf.get_params()
clf.set_params(n_jobs = 1)

'''
f = open("randomForest", "wb")
cPickle.dump(clf, f)
f.close()

f = open("randomForest", "r")
clf = cPickle.load(f)
'''

print rmse(X_test, y_test, clf.predict)
submission(clf.predict, filters)
