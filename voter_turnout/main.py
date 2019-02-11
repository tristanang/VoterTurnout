# From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network

import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
import normalize

import gradient_maps 

# Import data
file = open( "../data/train_input.pickle", "rb" )
X = pickle.load(file)
file.close

file = open( "../data/target.pickle", "rb" )
y = pickle.load(file)
file.close

# Separate validation and training set
nTrain = int(len(X) * 0.9)
X_train = X.iloc[0:nTrain, :]
X_val = X.iloc[len(X) - nTrain:, :]
y_train = y.iloc[0:nTrain]
y_val = y.iloc[len(y) - nTrain:]

# Normalization
# TODO change to colsToScale
colsToScale = gradient_maps.toNormalize
# Use this for all normalizations
scaler = normalize.retScaler(X_train, colsToScale)
normalize.scale(X_train, scaler, True)
normalize.scale(X_val, scaler, True)

# Save scaler
file = open("scaler.pickle", 'wb')
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
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators = 200, max_depth = 20, criterion='gini')

clf.fit(X_train, y_train)

train_acc = accuracy_score(clf.predict(X_train), y_train)
val_acc = accuracy_score(clf.predict(X_val), y_val)

# Save classifier
file = open("clf.pickle", 'wb')
pickle.dump(clf, file)
file.close()

print("train accuracy: ", train_acc, "test accuracy: ", val_acc)

# Save results
f = open("results.txt", 'w')
print("train accuracy: ", train_acc, "test accuracy: ", val_acc, file = f)
f.close()