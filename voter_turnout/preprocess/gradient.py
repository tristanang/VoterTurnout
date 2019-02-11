from voter_turnout import info
from voter_turnout.preprocess import gradient_maps
import numpy as np
import sys
import pandas as pd

master = gradient_maps.master

def main(df, master=master):
    for columnName in gradient_maps.mapOnly:
        print(columnName)
        df = mappingVals(df, columnName, gradient_maps.maps)

    for columnName in master.keys():
        print(columnName)
        df = oneHotNegatives(df, columnName)
        df = mappingVals(df, columnName)

    for columnName in gradient_maps.notDone:
        print(columnName)
        df = oneHotNegatives(df, columnName)

    return df

def oneHotNegatives(df, columnName):
    """
        Find negative values in columnName, one hots them and sets those values
        to zero.
    """
    oneHot = []

    for val in df[columnName].unique():
        if val < 0:
            oneHot.append(val)

    for negative in oneHot:
        column = str(columnName) + '_' + str(negative)
        
        try:
            df[column]
            sys.exit(column + ' already exists!')

        except KeyError:
            pass

        df[column] = np.where(df[columnName] == negative, 1, 0)
        df.loc[df[columnName] == negative, columnName] = 0
        print(df[columnName].unique())

    return df

def mappingVals(df, columnName, master=master):

    d = master[columnName]

    for k,v in iter(d.items()):
        df.loc[df[columnName] == k, columnName] = v

    return df

if __name__ == '__main__':
    filepath = 'data/afterIO.pickle'

    df = pd.read_pickle(filepath)

    df = main(df)

    print("This is the number of columns after gradienting: " + str(df.columns.size))

    df.to_pickle('data/afterGradient.pickle')
