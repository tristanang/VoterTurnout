from voter_turnout import info
from voter_turnout.preprocess import gradient_maps
import numpy as np
import sys

master = gradient_maps.master

def main(df, master):
    for columnName in master.keys():
        df = oneHotNegatives(df, columnName)
        df = mappingVals(df, columnName)

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

def mappingVals(df, columnName):

    d = master[columnName]

    for k,v in iter(d.items()):
        df.loc[df[columnName] == k, columnName] = v

    return df

