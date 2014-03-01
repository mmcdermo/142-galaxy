from util import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import cross_validation

filters = [crop((150, 275), (150, 275)), resize(36, 36), grayscale]
(X, Y) = loadProcess(0.03, filters)

preview1D(X[0], 36, 36)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, Y, test_size=0.4, random_state=0)

clf = RandomForestRegressor(n_estimators=5, n_jobs=-1)
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)  
print score
