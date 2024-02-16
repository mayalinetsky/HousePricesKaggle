"""
Simple functions for preprocessors
"""
import logging
import warnings

import pandas as pd
from constants import *

COLUMNS_TO_DROP_AT_END = []


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

    _fill_na_GarageYrBlt_w_YearBuilt(data)

    _convert_nan_to_str(data)

    _fill_na_LotFrontage_w_neib_median(data)

    _convert_month_year_to_cat(data)

    return data


def _convert_month_year_to_cat(data: pd.DataFrame):
    data.loc[:, MoSold] = data[MoSold].astype(str)

    data.loc[:, YrSold] = data[YrSold].astype(str)


def drop_known_columns(data: pd.DataFrame):
    data = data.copy()

    data = _drop_categorical_features_w_low_correlation_to_target(data)

    data = _drop_imbalanced_features(data)

    data = _drop_correlated_features(data)
    return data


def drop_globally_gathered_columns(data: pd.DataFrame):
    return data.drop(columns=COLUMNS_TO_DROP_AT_END, errors='ignore')


def _convert_nan_to_str(data: pd.DataFrame, inplace: bool = True):
    """
    preprocess features with NaN values that reflect 'None' and should not be discarded (should be counted)
    """
    # relevant_columns = [BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1,
    #                     FireplaceQu,
    #                     GarageType, GarageFinish, GarageQual, GarageCond,
    #                     PoolQC,
    #                     MasVnrType,
    #                     MiscFeature]

    columns_to_fill = data.select_dtypes(include='object').columns
    tmp = data.loc[:, columns_to_fill].fillna(value='None', inplace=inplace)

    if not inplace:
        data.loc[:, columns_to_fill] = tmp
        
    return data


def _fill_na_GarageYrBlt_w_YearBuilt(data: pd.DataFrame, inplace: bool = True):
    return data.loc[:, GarageYrBlt].fillna(data[YearBuilt], inplace=inplace)


def _fill_na_LotFrontage_w_neib_median(data: pd.DataFrame):
    def fill_w_median_or_zero(x):
        with warnings.catch_warnings():  # catching 'mean of an empty slice'
            warnings.simplefilter("error", category=RuntimeWarning)
            try:
                return x.fillna(x.median())
            except RuntimeWarning:
                return x.fillna(0)

    data.loc[:, LotFrontage] = data.groupby(Neighborhood)[LotFrontage].transform(fill_w_median_or_zero)
    return data


def _drop_categorical_features_w_low_correlation_to_target(data: pd.DataFrame):
    """
    Drop features that show low correlation to target (by indirect/manual impression)
    """
    cat_cols_uncor_w_target = [LotShape, LandContour, LotConfig,
                               LandSlope, Condition2, RoofMatl, BsmtExposure,
                               BsmtFinType1, BsmtFinType2, Electrical,
                               Functional, Fence, MiscFeature
                               ]
    COLUMNS_TO_DROP_AT_END.extend(cat_cols_uncor_w_target)
    return data


def _drop_correlated_features(data: pd.DataFrame):
    # In the data, '1stFlrSF' + '2ndFlrSF' = 'GrLivArea'
    # We dropped '1stFlrSF' due to high correlation with 'GrLivArea'
    COLUMNS_TO_DROP_AT_END.extend([FirststFlrSF])
    return data


def _drop_imbalanced_features(data: pd.DataFrame):
    imbalanced = [Heating, Alley, Street, Utilities]
    COLUMNS_TO_DROP_AT_END.extend(imbalanced)
    return data
