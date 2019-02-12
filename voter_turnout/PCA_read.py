import pickle
import pandas as pd

file = open( 'data/pca/pca_50.pickle', "rb" )
pca = pickle.load(file)
file.close

file = open( 'data/test_input.pickle', "rb" )
input_data = pickle.load(file)
file.close

components = pca.components_

components = pd.DataFrame(components, columns = input_data.columns.values)
components.columns

# Save results
components.to_csv('data/pca/pca_50_components.csv')

