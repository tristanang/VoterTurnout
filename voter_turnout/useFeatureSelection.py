import pickle

# Load selector
file = open( "data/lasso/lasso_select.pickle", "rb" )
model = pickle.load(file)
file.close


name = "data/train_input"

# Import data
file = open( "{}.pickle".format(name), "rb" )
X = pickle.load(file)
file.close

# Output which columns are kept
colsKept = model.transform([X.columns.values, ])

print(colsKept)
print(len(colsKept[0]))

# Save results
f = open("data/lasso/cols.txt", 'w')
print(colsKept[0], file = f)
f.close()