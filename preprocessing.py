"""
Simple functions for preprocessors
"""
import pandas as pd


def baseline_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    _fill_na_GarageYrBlt_w_YearBuilt(data)

    return data.select_dtypes(include='number')


def preprocess(data: pd.DataFrame):
    """
    Prepare the data before model fitting:
    1. Converting some nan values to str 'No'
    2. Convert 'PoolArea' and 'PoolQC' to one binary feature 'HavePool'
    """
    data = data.copy()

    _convert_nan_to_str(data)

    _convert_pool_features_to_binary(data)

    return data


def _convert_nan_to_str(data: pd.DataFrame, inplace: bool = True):
    """
    preprocess features with NaN values that reflect 'None' and should not be discarded (should be counted)
    """
    relevant_columns = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'PoolQC',
                        'MiscFeature']

    for feature in relevant_columns:
        try:
            return data[feature].fillna(value='No', inplace=inplace)
        except KeyError:
            pass


def _convert_pool_features_to_binary(data: pd.DataFrame, inplace: bool = True):
    """
    only 7 samples with pool, but might be important, so:
    we create new *binary* feature 'HavePool' and drop 'PoolQC' 'PoolArea'
    """
    data.loc[data['PoolArea'] != 0, 'HavePool'] = 1
    data.loc[data['PoolArea'] == 0, 'HavePool'] = 0
    return data.drop(['PoolArea', 'PoolQC'], axis=1, inplace=inplace)


def _fill_na_GarageYrBlt_w_YearBuilt(data: pd.DataFrame, inplace: bool = True):
    return data['GarageYrBlt'].fillna(data['YearBuilt'], inplace=inplace)
