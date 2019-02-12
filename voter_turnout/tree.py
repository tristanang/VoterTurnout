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


def boost(X, y, params = None):
    if params is None:
        clf = AdaBoostClassifier(n_estimators=100, base_estimator=ExtraTreesClassifier(max_depth=5, n_estimators=10, n_jobs=-1), learning_rate=0.3)
        #clf = AdaBoostClassifier(n_estimators=147, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=-1), learning_rate=0.38979591836734695)
    else:
        clf = AdaBoostClassifier(**params)
    clf.fit(X, y)
    return clf

'''
{'n_estimators': 147, 'learning_rate': 0.38979591836734695, 'base_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)}
[{'n_estimators': 109, 'learning_rate': 0.3061224489802857, \
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
'''
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