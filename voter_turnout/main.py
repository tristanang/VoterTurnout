

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.feature_selection import SelectFromModel

start = datetime.datetime.now()
print(datetime.datetime.now())


# Save path
save_path = "models/"

# Import data
file = open( "data/train_input.pickle", "rb" )
X = pickle.load(file)
X = np.array(X)
file.close

file = open( "data/target.pickle", "rb" )
y = pickle.load(file)
file.close

# Separate validation and training set
nTrain = int(len(y) * 1)
skf = StratifiedKFold(n_splits=5)
train_aucs, val_aucs = [], []
#for train_index, test_index in skf.split(X, y):
for train_index, test_index in [[list(range(nTrain)), list(range(nTrain, len(y)))],]:
    X_train = X[train_index]
    X_val = X[test_index]
    y_train = y.iloc[train_index]
    y_val = y.iloc[test_index]
    
    
    # Train a model
    from voter_turnout import tree
    clf = tree.boost(X_train, y_train)
    
    # Classification accuracy
    train_acc = accuracy_score(clf.predict(X_train), y_train)
    #val_acc = accuracy_score(clf.predict(X_val), y_val)
    
    # Area under the ROC curve
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    #val_auc =  roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    
    train_aucs.append(train_auc)
    #val_aucs.append(val_auc)
    
    print()
    #print("train accuracy: ", train_acc, "test accuracy: ", val_acc)
    #print("train AUC: ", train_auc, "test AUC: ", val_auc)
    #print("ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))
    
    # Save classifier
    file = open(save_path + "clf.pickle", 'wb')
    pickle.dump(clf, file)
    file.close()
    '''
    # Save results
    f = open(save_path + "results.txt", 'w')
    print("train accuracy: ", train_acc, " test accuracy: ", val_acc, \
          "\ntrain AUC: ", train_auc, " test AUC: ", val_auc, \
          "\nave train AUC: ", np.mean(train_aucs), " ave test AUC: ", np.mean(val_aucs), \
          file = f)
    f.close()
    '''
    
#print("FINAL ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))

print(datetime.datetime.now())
print("time elapsed: ", datetime.datetime.now() - start)