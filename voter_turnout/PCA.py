# PCA
# From https://www.kaggle.com/pmmilewski/pca-decomposition-and-keras-neural-network

pca = PCA(n_components=1000)
pca.fit(X_train)

# Visualize how many components needed
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

plt.show()

# Re-run PCA for components wanted
NCOMPONENTS = 100

pca = PCA(n_components=NCOMPONENTS)
X_train = pca.fit_transform(X_train)
#X_val = pca.transform(X_val)
pca_std = np.std(X_pca_train)