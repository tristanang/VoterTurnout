# From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network

import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from voter_turnout import normalize

from voter_turnout.preprocess import gradient_maps 

# Save path
save_path = "models/"

# Import data
file = open( "data/train_input.pickle", "rb" )
X = pickle.load(file)
file.close

file = open( "data/target.pickle", "rb" )
y = pickle.load(file)
file.close

# Separate validation and training set
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    nTrain = int(len(X) * 0.9)
    X_train = X.iloc[train_index, :]
    X_val = X.iloc[test_index, :]
    y_train = y.iloc[train_index]
    y_val = y.iloc[test_index]
    
    # Normalization
    # TODO change to colsToScale
    colsToScale = gradient_maps.toNormalize
    # Use this for all normalizations
    scaler = normalize.retScaler(X_train, colsToScale)
    normalize.scale(X_train, scaler, True)
    normalize.scale(X_val, scaler, True)
    
    # Save scaler
    file = open(save_path + "scaler.pickle", 'wb')
    pickle.dump(scaler, file)
    file.close()
    
    
    
    # PCA
    '''pca = PCA(n_components=500)
    pca.fit(X_train)
    
    # Visualize how many components needed
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    
    
    # Re-run PCA for components wanted
    NCOMPONENTS = 100
    
    pca = PCA(n_components=NCOMPONENTS)
    X_pca_train = pca.fit_transform(X_sc_train)
    #X_pca_test = pca.transform(X_sc_test)
    pca_std = np.std(X_pca_train)
    '''
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    clf = RandomForestClassifier(n_estimators = 100, max_depth = 20, criterion='gini')
    
    clf.fit(X_train, y_train)
    
    # Classification accuracy
    train_acc = accuracy_score(clf.predict(X_train), y_train)
    val_acc = accuracy_score(clf.predict(X_val), y_val)
    
    # Area under the ROC curve
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    val_auc =  roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    
    # Save classifier
    file = open(save_path + "clf.pickle", 'wb')
    pickle.dump(clf, file)
    file.close()
    
    print()
    print("train accuracy: ", train_acc, "test accuracy: ", val_acc)
    print("train AUC: ", train_auc, "test AUC: ", val_auc)
    
    # Save results
    f = open(save_path + "results.txt", 'w')
    print("train accuracy: ", train_acc, "test accuracy: ", val_acc, \
          "\ntrain AUC: ", train_auc, "test AUC: ", val_auc, \
          file = f)
    f.close()