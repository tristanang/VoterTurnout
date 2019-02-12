

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel



# Save path
save_path = "data/adaboost regularized 2012"
name = "data/for_2012_train_set_v1"

# Import data
file = open( "{}.pickle".format(name), "rb" )
X = pickle.load(file)
X_train = np.array(X)
file.close

file = open( "data/target.pickle", "rb" )
y_train = pickle.load(file)
file.close
    
    
# Train a model
from voter_turnout import tree
clf = tree.forest(X_train, y_train,
                  n_estimators = 300, max_depth = 15, min_samples_leaf = 30)

# Get feature selection
model = SelectFromModel(clf, prefit=True)


# Save classifier
file = open(save_path + "clf.pickle", 'wb')
pickle.dump(clf, file)
file.close()

# Save selecor
file = open(save_path + "select_feature.pickle", 'wb')
pickle.dump(model, file)
file.close()

# Transform data
X_transf = model.transform(X_train)

print(len(X_transf[0]))

# Save transformed data
file = open('{}_some_features.pickle'.format(name), 'wb')    
pickle.dump(X_transf, file)
file.close()


# Output which columns are kept
colsKept = model.transform([X.columns.values, ])
# Save results
f = open(save_path + "cols_kept.txt", 'w')
print(colsKept[0], file = f)
f.close()