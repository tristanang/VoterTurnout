import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from voter_turnout import normalize
from voter_turnout.preprocess import gradient_maps 

def loadData():
    # Import data
    file = open( "data/train_input.pickle", "rb" )
    X = pickle.load(file)
    file.close

    file = open( "data/target.pickle", "rb" )
    y = pickle.load(file)
    file.close

    # Normalization
    colsToScale = gradient_maps.toNormalize
    # Use this for all normalizations
    scaler = normalize.retScaler(X, colsToScale)
    normalize.scale(X, scaler, True)

    # Save scaler
    file = open(save_path + "scaler.pickle", 'wb')
    pickle.dump(scaler, file)
    file.close()  

    return X, y

# Save path
save_path = "models/"

# model for swapping
model = XGBClassifier(n_jobs=-1)
params = {}


if __name__ == '__main__':
    X, y = loadData()

    param_grid = {"n_estimators": [5, 10, 20, 30, 40, 50], "base_estimator": [RandomForestClassifier(max_depth=1, n_estimators=10), RandomForestClassifier(max_depth=2, n_estimators=10)],}

    gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, 
                                            cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True,)

    gridSearch.fit(X, y)

    print(gridSearch.best_score_)
    print(gridSearch.best_params_)
    print(gridSearch.best_index_)

    pickle.dump(gridSearch.cv_results_, open("grid_ada.pickle", 'wb'))



