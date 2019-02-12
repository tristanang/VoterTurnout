
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Path with model info
path = "models/"


# Import feature selection
file = open( "data/selectFeature/select_feature.pickle", "rb" )
feature = pickle.load(file)
file.close

# Import input set
file = open( "data/test_input.pickle", "rb" )
X = pickle.load(file)
file.close

# Transform
#X = feature.transform(X)


# Import classifier
file = open( path + "clf.pickle", "rb" )
clf = pickle.load(file)
file.close

# Predict
# Probability
y_predict = clf.predict_proba(X)[:, 1]

# Output format
arr = pd.DataFrame(y_predict)
arr.columns = ["target"]

# Export
arr.to_csv( path + "predicted_target.csv")
