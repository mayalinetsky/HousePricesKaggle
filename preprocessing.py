"""
Module to prepare the dataframe to model fitting
"""
import pandas as pd


def preprocess(data: pd.DataFrame):
    """
    Prepare the data before model fitting:
    1. Converting some nan values to str
    2. Convert 'PoolArea' and 'PoolQC' to one binary feature 'HavePool'
    """
    data = data.copy()

    _convert_nan_to_str(data)

    _convert_pool_features_to_binary(data)

    _fill_na_lot_frontage(data)
    return data


def _convert_nan_to_str(data: pd.DataFrame):
    """
    preprocess features with NaN values that reflect 'None' and should not be discarded (should be counted)
    """
    relevant_columns = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'PoolQC',
                        'MiscFeature']

    for feature in relevant_columns:
        try:
            data[feature].fillna(value='No', inplace=True)
        except KeyError:
            pass


def _convert_pool_features_to_binary(data: pd.DataFrame):
    """
    only 7 samples with pool, but might be important, so:
    we create new *binary* feature 'HavePool' and drop 'PoolQC' 'PoolArea'
    """
    data.loc[data['PoolArea'] != 0, 'HavePool'] = 1
    data.loc[data['PoolArea'] == 0, 'HavePool'] = 0
    data.drop(['PoolArea', 'PoolQC'], axis=1, inplace=True)


def _fill_na_lot_frontage(data: pd.DataFrame):
    """

    """
    mean_value_LotFrontage = data['LotFrontage'].mean()
    data['LotFrontage'].fillna(value=mean_value_LotFrontage, inplace=True)
