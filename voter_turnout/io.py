import pandas as pd
import numpy as np

def readFile(path):
    df = pd.read_csv(path)

    toDrop = ['id']

    df.drop(columns=toDrop)

    return df