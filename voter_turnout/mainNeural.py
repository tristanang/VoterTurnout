

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

 

# Save path
save_path = "models/"

# Import data
file = open( "data/train_input_norm_PCA500.pickle", "rb" )
X = pickle.load(file)
file.close

file = open( "data/target.pickle", "rb" )
y = pickle.load(file)
y = np.array(y)
file.close

print([val for val in np.max(X, axis=0) if val > 10])

exit()
# Separate validation and training set
nTrain = int(len(y) * 0.9)
skf = StratifiedKFold(n_splits=3)
train_aucs, val_aucs = [], []
for train_index, test_index in skf.split(X, y):
#for train_index, test_index in [[list(range(nTrain)), list(range(nTrain, len(y)))],]:
    X_train = X[train_index]
    X_val = X[test_index]
    y_train = y[train_index]
    y_val = y[test_index] 
    
    
    # Train a model
    from voter_turnout import neuralNet
    clf = neuralNet.neuralNet(X_train, y_train, X_val, y_val)
    
    # Area under the ROC curve
    train_auc = roc_auc_score(y_train, clf.predict(X_train))
    val_auc =  roc_auc_score(y_val, clf.predict(X_val))
    
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    
    print()
    print("train AUC: ", train_auc, "test AUC: ", val_auc)
    print("ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))
    
    # Save classifier
    file = open(save_path + "clf.pickle", 'wb')
    pickle.dump(clf, file)
    file.close()
    
    # Save results
    f = open(save_path + "results.txt", 'w')
    print(\
          "\ntrain AUC: ", train_auc, "test AUC: ", val_auc, \
          "ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs), \
          file = f)
    f.close()
    
print("FINAL ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))

