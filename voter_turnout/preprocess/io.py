import pandas as pd
import numpy as np
from voter_turnout import (info)

def readFile(path, toDrop=info.toDrop):
    """ 
        Reads the csv file and drops some columns which are useless as defined in
        20008-Codebook
    """
    df = pd.read_csv(path)

    assert (df.columns.size - len(toDrop)) == df.drop(columns=toDrop).columns.size

    df = df.drop(columns=toDrop)

    return df

def dropSameColumn(df):
    """
        Some columns have values that are all identical. We drop all of those.
    """
    new_df = df

    for column in df.columns:
        if df[column].unique().size == 1:
            new_df = new_df.drop(columns=column)

    return new_df

def removeBlanks(df):
    """
        In the dataset, some values are -1, which are blank. This function is to
        remove them so they are not one-hot encoded.
    """
    return df

def oneHot(df, noHot=info.noOneHot):
    oneHot = []

    for column in df.columns:
        if column not in info.noOneHot:
            oneHot.append(column)

    for column in oneHot:
        tempHot = pd.get_dummies(df[column],prefix=column)
        df = df.drop(columns=column)
        df = df.join(tempHot)

    return df


def separateInput(filename = "data/oneHot.pickle", targetName='target'):
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







