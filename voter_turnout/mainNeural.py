

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from voter_turnout import normalize

from voter_turnout.preprocess import gradient_maps 

# Save path
save_path = "models/"

# Import data
file = open( "data/train_input_norm_PCA500.pickle", "rb" )
X = pickle.load(file)
file.close

file = open( "data/target.pickle", "rb" )
y = pickle.load(file)
file.close

# Separate validation and training set
nTrain = int(len(y) * 0.9)
skf = StratifiedKFold(n_splits=5)
train_aucs, val_aucs = [], []
#for train_index, test_index in skf.split(X, y):
for train_index, test_index in [[list(range(nTrain)), list(range(nTrain, len(y)))],]:
    X_train = X[train_index]
    X_val = X[test_index]
    y_train = y.iloc[train_index]
    y_val = y.iloc[test_index] 
    
    
    # Train a model
    from voter_turnout import neuralNet
    clf = neuralNet.neuralNet(X_train, y_train, X_val, y_val)
    
    ## Printing the accuracy of our model, according to the loss function specified in model.compile above
    score = model.evaluate(X_test, y_test, verbose=0)
    in_sample_score = model.evaluate(X_train, y_train, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('In-sample accuracy:', in_sample_score[1])

    
    # Classification accuracy
    #train_acc = accuracy_score(clf.predict(X_train), y_train)
    #val_acc = accuracy_score(clf.predict(X_val), y_val)
    
    # Area under the ROC curve
    train_auc = roc_auc_score(y_train, clf.predict(X_train)[:, 1])
    val_auc =  roc_auc_score(y_val, clf.predict(X_val)[:, 1])
    
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    
    print()
    #print("train accuracy: ", train_acc, "test accuracy: ", val_acc)
    print("train AUC: ", train_auc, "test AUC: ", val_auc)
    print("ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))
    
    # Save classifier
    file = open(save_path + "clf.pickle", 'wb')
    pickle.dump(clf, file)
    file.close()    
    
    # Save results
    f = open(save_path + "results.txt", 'w')
     #"train accuracy: ", train_acc, "test accuracy: ", val_acc, \
    print(\
          "\ntrain AUC: ", train_auc, "test AUC: ", val_auc, \
          file = f)
    f.close()
    
print("FINAL ave train AUC: ", np.mean(train_aucs), "ave test AUC: ", np.mean(val_aucs))

