"""
Simple functions for preprocessors
"""
import logging

import pandas as pd
from constants import *
from utils import calc_numeric_feature_correlation


def baseline_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    _fill_na_GarageYrBlt_w_YearBuilt(data)

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


def drop_columns(data: pd.DataFrame):
    data = data.copy()

    data = _drop_highly_correlated_numeric_features(data)

    data = _drop_categorical_features_w_low_correlation_to_target(data)

    data = _drop_imbalanced_features(data)

    data = _drop_correlated_features(data)
    return data


def _convert_nan_to_str(data: pd.DataFrame, inplace: bool = True):
    """
    preprocess features with NaN values that reflect 'None' and should not be discarded (should be counted)
    """
    relevant_columns = [BsmtQual, BsmtCond, FireplaceQu, GarageType,
                        GarageFinish, GarageQual, GarageCond, BsmtExposure, BsmtFinType1, PoolQC,
                        MiscFeature]

    for feature in relevant_columns:
        try:
            tmp = data[feature].fillna(value='None', inplace=inplace)
            if not inplace:
                data[feature] = tmp
        except KeyError:
            pass
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


def _drop_highly_correlated_numeric_features(data: pd.DataFrame, threshold: float = 0.7):
    """
    Calculates the correlation between all pairs of numeric features in the dataframe, looks at all the pairs
    with correlation above the threshold, and drops the second feature in these pairs.
    """
    numeric_correlations = calc_numeric_feature_correlation(data)

    highly_correlated_numeric_features = [t for t in numeric_correlations if t[2] >= threshold]

    high_correlated_features_to_drop = [t[1] for t in highly_correlated_numeric_features]

    logging.debug(f"Dropping the following highly correlated numeric features: {high_correlated_features_to_drop}")

    return data.drop(columns=high_correlated_features_to_drop, errors='ignore')


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
