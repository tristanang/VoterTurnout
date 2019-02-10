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