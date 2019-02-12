import pickle

from voter_turnout import normalize
from voter_turnout.preprocess import gradient_maps


# Load data
name = "data/train_input"
file = open( "{}.pickle".format(name), "rb" )
X = pickle.load(file)
file.close


# Normalization
colsToScale = gradient_maps.toNormalize
# Use this for all normalizations
scaler = normalize.retScaler(X, colsToScale)
normalize.scale(X, scaler, True)
# Save scaler
file = open("data/scaler.pickle", 'wb')
pickle.dump(scaler, file)
file.close()


# Select features
# Load selector
file = open( "data/selectFeature/select_feature.pickle", "rb" )
model = pickle.load(file)
file.close

# Transform data
X_transf = model.transform(X)

# Save transformed data
file = open('{}_some_features_normed.pickle'.format(name), 'wb')
pickle.dump(X_transf, file)
file.close()


'''
# Output which columns are kept
colsKept = model.transform([X.columns.values, ])

print(colsKept)
print(len(colsKept[0]))

# Save results
f = open("data/lasso/cols.txt", 'w')
print(colsKept[0], file = f)
f.close()
'''