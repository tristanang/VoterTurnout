# Code from set 3

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators = 100, max_depth = 20, criterion='gini')

clf.fit(X_train, y_train)

train_acc = accuracy_score(clf.predict(X_train), y_train)
val_acc = accuracy_score(clf.predict(X_val), y_val)

print("train error: ", train_acc, "test error: ", val_acc)