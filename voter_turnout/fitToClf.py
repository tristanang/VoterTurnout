
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Custom
from voter_turnout import normalize

# Path with model info
path = "models/forest, 200, 20/"

# Import scaler, classifier
file = open( path + "scaler.pickle", "rb" )
scaler = pickle.load(file)
file.close

file = open( path + "clf.pickle", "rb" )
clf = pickle.load(file)
file.close


# Import input set
file = open( "data/test_input.pickle", "rb" )
X = pickle.load(file)
file.close


# Normalize
normalize.scale(X, scaler, True)

# Predict
# Probability
y_predict = clf.predict_proba(X)[:, 1]

# Output format
arr = pd.DataFrame(y_predict)
arr.columns = ["target"]

# Export
arr.to_csv(path + "predicted_target.csv")
