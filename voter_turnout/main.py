# From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import info

# Import data
file = open( "../data/input.pickle", "rb" )
X = pickle.load(file)
file.close

# Separate
nTrain = int(len(X) * 0.1)
X_train = X[0:nTrain]
X_val = X[len(X) - nTrain:]


# Normalization
# Scale columns
def retScaler(trainingSet, colsToScale):
    '''
    Returns a scaler for the specified columns and training set.
    Can scale a dataset using scale(dataset, return val of this function).
    '''
    scaler = StandardScaler()
    scaler.fit(trainingSet[colsToScale])
    return (scaler, colsToScale)


def scale(dataset, customScaler):
    '''
    Returns a copy of dataset with columns scaled as setup by customScaler.
    '''
    scaler, colsToScale = customScaler
    newSet = dataset.copy()
    newSet[colsToScale] = scaler.transform(dataset[colsToScale])
    return newSet

# TODO change to colsToScale
colsToScale = info.gradient
scaler = retScaler(X_train, colsToScale)
X_sc_train = scale(X_train, scaler)
#X_sc_test = scale(X_val, scaler)

# PCA
pca = PCA(n_components=500)
pca.fit(X_sc_train)

# Visualize how many components needed
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# Re-run PCA for components wanted
NCOMPONENTS = 100

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(X_sc_train)
X_pca_test = pca.transform(X_sc_test)
pca_std = np.std(X_pca_train)

