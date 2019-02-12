

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from voter_turnout import tree

start = datetime.datetime.now()
print(datetime.datetime.now())


# Save path
save_path = "models/"

# Import data
file = open( "data/for_2012_train_set_v1_some_features.pickle", "rb" )
X = pickle.load(file)
X_train = np.array(X)
file.close

file = open( "data/target.pickle", "rb" )
y_train = pickle.load(file)
file.close

# Input params
params = [None, ]

for i, param in enumerate(params):
    print(param)
    save_path = "models/adaboost regularized 2012/"
    
    # Train a model
    clf = tree.boost(X_train, y_train, param)
    
    # Classification accuracy
    #train_acc = accuracy_score(clf.predict(X_train), y_train)
    
    
    # Area under the ROC curve
    #train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    
    
    # Save classifier
    file = open(save_path + "clf.pickle", 'wb')
    pickle.dump(clf, file)
    file.close()
    
    # Import input set
    file = open( "data/for_2012_test_set_v1.pickle", "rb" )
    X_test = pickle.load(file)
    file.close
    
    
    # Import feature selection
    file = open( "data/selectFeature 2012/select_feature.pickle", "rb" )
    feature = pickle.load(file)
    file.close
    # Transform
    X_test = feature.transform(X_test)
    
    
    # Predict
    # Probability
    y_predict = clf.predict_proba(X_test)[:, 1]
    
    # Output format
    arr = pd.DataFrame(y_predict)
    arr.columns = ["target"]
    
    # Export
    arr.to_csv(save_path + "predicted_target.csv")
    
    
    # Save results
    f = open(save_path + "results.txt", 'w')
    print("done", \
          file = f)
    f.close()
    
print(datetime.datetime.now())
print("time elapsed: ", datetime.datetime.now() - start)