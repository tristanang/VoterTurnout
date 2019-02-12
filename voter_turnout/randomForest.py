import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

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

    # PCA
    # From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network
    # Decided by trying more, looking at graph. See PCA.py
    NCOMPONENTS = 75
    
    pca = PCA(n_components=NCOMPONENTS)
    X = pca.fit_transform(X)
    #pca_std = np.std(X_pca_train)
    
    # Save PCA
    file = open(save_path + "pca.pickle", 'wb')
    pickle.dump(scaler, file)
    file.close()    

    return X, y

# Save path
save_path = "models/"

# model for swapping
model = RandomForestClassifier()
params = {}


if __name__ == '__main__':
    X, y = loadData()

    param_grid = {"n_estimators": np.arange(1,500), "max_depth": np.arange(1,30), "min_samples_leaf": np.arange(1,300),\
                  "criterion": ["gini", "entropy"],}

    for i in range(20):
        print(i)

        randomSearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10,\
                                            cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True,)
        try:
            randomSearch.fit(X, y)

            
            print(randomSearch.best_score_)
            print(randomSearch.best_params_)
            print(randomSearch.best_index_)

            pickle.dump(randomSearch.cv_results_, open("random{}.pickle".format(i), 'wb'))
        except:
            print("fuck")


