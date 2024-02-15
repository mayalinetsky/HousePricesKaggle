"""
Simple functions for preprocessors
"""
import logging

import pandas as pd
from constants import *


def baseline_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # _fill_na_GarageYrBlt_w_YearBuilt(data)

    return data.select_dtypes(include='number')


def preprocess(data: pd.DataFrame):
    """
    Prepare the data before model fitting:
    1. Converting some nan values to str 'None'
    2. Convert 'PoolArea' and 'PoolQC' to one binary feature 'HavePool'
    """
    data = data.copy()

    _convert_nan_to_str(data)

    _convert_pool_features_to_binary(data)

    return data


def drop_known_columns(data: pd.DataFrame):
    data = data.copy()

    data = _drop_categorical_features_w_low_correlation_to_target(data)

    data = _drop_imbalanced_features(data)

    data = _drop_correlated_features(data)
    return data


def _convert_nan_to_str(data: pd.DataFrame, inplace: bool = True):
    """
    preprocess features with NaN values that reflect 'None' and should not be discarded (should be counted)
    """
    relevant_columns = [BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1,
                        FireplaceQu,
                        GarageType, GarageFinish, GarageQual, GarageCond,
                        PoolQC,
                        MasVnrType,
                        MiscFeature]

    columns_to_fill = list(set(relevant_columns) & set(data.columns))
    tmp = data.loc[:, columns_to_fill].fillna(value='None', inplace=inplace)

    if not inplace:
        data.loc[:, columns_to_fill] = tmp
        
    return data


def _convert_pool_features_to_binary(data: pd.DataFrame, inplace: bool = True):
    """
    only 7 samples with pool, but might be important, so:
    we create new *binary* feature 'HavePool' and drop 'PoolQC' 'PoolArea'
    """
    data.loc[data['PoolArea'] != 0, 'HavePool'] = 1
    data.loc[data['PoolArea'] == 0, 'HavePool'] = 0
    return data.drop(['PoolArea', 'PoolQC'], axis=1, errors='ignore', inplace=inplace)


def _fill_na_GarageYrBlt_w_YearBuilt(data: pd.DataFrame, inplace: bool = True):
    return data['GarageYrBlt'].fillna(data['YearBuilt'], inplace=inplace)


def _drop_categorical_features_w_low_correlation_to_target(data: pd.DataFrame):
    """
    Drop features that show low correlation to target (by indirect/manual impression)
    """
    cat_cols_uncor_w_target = [LotShape, LandContour, LotConfig,
                               LandSlope, Condition2, RoofMatl, BsmtExposure,
                               BsmtFinType1, BsmtFinType2, Electrical,
                               Functional, Fence, MiscFeature
                               ]
    return data.drop(columns=cat_cols_uncor_w_target, errors='ignore')


def _drop_correlated_features(data: pd.DataFrame):
    # In the data, '1stFlrSF' + '2ndFlrSF' = 'GrLivArea'
    # We dropped '1stFlrSF' due to high correlation with 'GrLivArea'
    return data.drop(columns=[FirststFlrSF], errors='ignore')


def _drop_imbalanced_features(data: pd.DataFrame):
    imbalanced = [Heating, Alley, Street, Utilities]
    return data.drop(columns=imbalanced, errors='ignore')
