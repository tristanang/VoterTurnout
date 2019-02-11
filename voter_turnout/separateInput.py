import pickle
import numpy as np

def separateInput(filename = "../data/oneHot.pickle", targetName='target'):
    # Load pickled df
    file = open( filename, "rb" )
    df = pickle.load(file)
    file.close
    
    # Separate
    X = df.drop(targetName, axis=1)
    y = df[targetName]    
    
    # Store
    X.to_pickle('input.pickle')
    y.to_pickle('target.pickle')

separateInput()