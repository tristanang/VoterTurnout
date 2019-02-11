from sklearn.preprocessing import StandardScaler

# Supress conversion warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Supress pandas dataframe copy warning
# Supress conversion warnings
from pandas.exceptions import SettingWithCopyWarning
warnings.filterwarnings(action='ignore', category=SettingWithCopyWarning)

# Scale columns

def retScaler(trainingSet, colsToScale):
    '''
    Returns a scaler for the specified columns and training set.
    Can scale a dataset using scale(dataset, return val of this function).
    '''
    scaler = StandardScaler()
    scaler.fit(trainingSet.loc[:, colsToScale])
    return (scaler, colsToScale)


def scale(dataset, customScaler, inPlace = False):
    '''
    Returns a copy of dataset with columns scaled as setup by customScaler.
    '''
    scaler, colsToScale = customScaler
    if inPlace:
        dataset.loc[:, colsToScale] = scaler.transform(dataset.loc[:, colsToScale])
        return dataset
    else:
        newSet = dataset.copy()
        #newSet.loc[:, colsToScale] = scaler.transform(dataset.loc[:, colsToScale])
        return newSet