# Code from set 3

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

def forest(X, y, n_estimators = 100, max_depth = 15, min_samples_leaf = 5):
    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth, 
                                 min_samples_leaf = min_samples_leaf,
                                 criterion='gini')
    
    clf.fit(X, y)
    
    return clf

def extra_random_forest(X, y, n_estimators = 100, max_depth = 15, min_samples_leaf = 5):
    clf = ExtraTreesClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth, 
                                 min_samples_leaf = min_samples_leaf,
                                 criterion='gini')
    
    clf.fit(X, y)
    
    return clf


def boost(X, y):
    clf = AdaBoostClassifier(n_estimators=200, base_estimator=ExtraTreesClassifier(max_depth=4, n_estimators=10, n_jobs=-1), learning_rate=0.7)
    clf.fit(X, y)
    return clf

'''
Implementation in main:

from voter_turnout import tree
clf = tree.forest(X_train, y_train,
                  n_estimators = 100, max_depth = 15, min_samples_leaf = 5)

# Classification accuracy
train_acc = accuracy_score(clf.predict(X_train), y_train)
val_acc = accuracy_score(clf.predict(X_val), y_val)

# Area under the ROC curve
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
val_auc =  roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

train_aucs.append(train_auc)
val_aucs.append(val_auc)

print()
print("train accuracy: ", train_acc, "test accuracy: ", val_acc)
print("train AUC: ", train_auc, "test AUC: ", val_auc)
print("ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))

# Save classifier
file = open(save_path + "clf.pickle", 'wb')
pickle.dump(clf, file)
file.close()    

# Save results
f = open(save_path + "results.txt", 'w')
print("train accuracy: ", train_acc, "test accuracy: ", val_acc, \
      "\ntrain AUC: ", train_auc, "test AUC: ", val_auc, \
      file = f)
f.close()

'''