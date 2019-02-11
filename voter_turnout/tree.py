# Code from set 3

from sklearn.ensemble import RandomForestClassifier

clf = tree.DecisionTreeClassifier(max_depth = 20, criterion='gini')

clf.fit(X_train, y_train)

train_err = classification_err(clf.predict(X_train), y_train)
test_err = classification_err(clf.predict(X_val), y_val)

print("train error: ", train_error, "test error: ", test_err)