

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
file = open( "data/train_input.pickle", "rb" )
X = pickle.load(file)
X_train = np.array(X)
file.close

file = open( "data/target.pickle", "rb" )
y_train = pickle.load(file)
file.close

# Input params
params = [{'n_estimators': 109, 'learning_rate': 0.3061224489802857, \
'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=8, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
    {'n_estimators': 151, 'learning_rate': 0.36734693877614283, \
    'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=6, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
            {'n_estimators': 145, 'learning_rate': 0.142857142858, \
            'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=7, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
            {'n_estimators': 194, 'learning_rate': 0.12244897959271428, \
            'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=7, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
            {'n_estimators': 108, 'learning_rate': 0.2612244897959184, \
            'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', \
            max_depth=9, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
            {'n_estimators': 160, 'learning_rate': 0.1602040816326531,\
            'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=8, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}, \
            {'n_estimators': 136, 'learning_rate': 0.24285714285714288, \
            'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\
            max_depth=8, max_features='auto', max_leaf_nodes=None,\
            min_impurity_decrease=0.0, min_impurity_split=None,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0,\
            warm_start=False)}]

for i, param in enumerate(params):
    print(param)
    save_path = "models/adaboost " + str(i) + "/"
    
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
    file = open( "data/test_input.pickle", "rb" )
    X_test = pickle.load(file)
    file.close
    
    '''
    # Import feature selection
    file = open( "data/selectFeature/select_feature.pickle", "rb" )
    feature = pickle.load(file)
    file.close
    # Transform
    X_test = feature.transform(X_test)
    '''
    
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