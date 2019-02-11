import pickle

file = open( 'data/pca/pca_100.pickle', "rb" )
pca = pickle.load(file)
file.close

print(pca)