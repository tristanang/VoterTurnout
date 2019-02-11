# Do normalization and PCA
# From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network

import pickle
from sklearn.decomposition import PCA

from voter_turnout import normalize
from voter_turnout.preprocess import gradient_maps 

# Load data
save_path = "data/pca"
name = "data/train_input"
file = open( '{}.pickle'.format(name), "rb" )
X = pickle.load(file)
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

# Do PCA
for n_components in [50, 75, 100, 200, 500, 1000]:
    
    
    pca = PCA(n_components=n_components)
    X_transf = pca.fit_transform(X)

    # Save PCA
    print('{}/pca_{}.pickle'.format(save_path, n_components))
    file = open('{}/pca_{}.pickle'.format(save_path, n_components), 'wb')
    pickle.dump(scaler, file)
    file.close()  
    
    # Save transformed data
    file = open('{}_norm_PCA{}.pickle'.format(name, n_components), 'wb')    
    pickle.dump(X_transf, file)
    file.close()
    


def chooseN_components():
    pca = PCA(n_components=1000)
    pca.fit(X_train)
    
    # Visualize how many components needed
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    
    plt.show()    