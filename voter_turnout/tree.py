# Code from set 3

from sklearn.ensemble import RandomForestClassifier

def forest(X, y, n_estimators = 100, max_depth = 15, min_samples_leaf = 5):
    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                 max_depth = max_depth, 
                                 min_samples_leaf = min_samples_leaf,
                                 criterion='gini')
    
    clf.fit(X, y)
    
    return clf