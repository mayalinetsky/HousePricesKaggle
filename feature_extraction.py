"""
Module encapsulating "feature extraction" phase that happens after splitting to folds.
It gets raw data and transforms it into features ready for preprocessing and labeling.
"""
import logging
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from constant_extracted import *
from constants import *
from utils import calc_numeric_feature_correlation
from preprocessing import COLUMNS_TO_DROP_AT_END


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Base class for transforming raw data into new features.

    For example: one-hot encoding
    """

    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def set_output(self, *, transform=None):
        pass


def join_porch_areas(x: pd.DataFrame):
    """
    Adds the following new features to the dataframe (and drops the columns used for extraction):

    - 'TotClosePorchSF' the sum of 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'
    - 'TotOpenPorchSF' the sum of 'OpenPorchSF', 'WoodDeckSF'
    """
    x = x.copy()
    close_porches = [EnclosedPorch, ThreeSsnPorch, ScreenPorch]
    open_porches = [OpenPorchSF, WoodDeckSF]

    def join_areas(cols):
        return cols.sum()

    x[TotClosePorchSF] = x[close_porches].apply(join_areas, axis=1)
    x[TotOpenPorchSF] = x[open_porches].apply(join_areas, axis=1)

    COLUMNS_TO_DROP_AT_END.extend(close_porches + open_porches)
    return x


def join_liv_bsmt_areas(x: pd.DataFrame):
    """
    Summing total 'GrLivArea' and 'TotalBsmtSF' and produces 'TotalArea',
    and dropping the following features:
    - 'GrLivArea'
    - 'TotalBsmtSF'
    - 'BsmtFinSF2',
    - '2ndFlrSF'
    """
    x[TotalArea] = x[GrLivArea] + x[TotalBsmtSF]

    # columns_to_drop = [TotalBsmtSF, FirststFlrSF, BsmtFinSF2, SecondFlrSF]
    #
    # COLUMNS_TO_DROP_AT_END.extend(columns_to_drop)

    return x


def extract_LotAreaRemainder(x: pd.DataFrame):
    x[LotAreaRemainder] = x[LotArea] - x[FirststFlrSF] - x[GarageArea] - x[TotOpenPorchSF] - x[TotClosePorchSF]\
                          - x[MasVnrArea] - x[WoodDeckSF] - x[PoolArea]

    COLUMNS_TO_DROP_AT_END.extend([LotArea])

    return x


def binarize_pool(data: pd.DataFrame):
    """
    only 7 samples with pool, but might be important, so:
    we create new *binary* feature 'HavePool' and drop 'PoolQC' 'PoolArea'
    """
    data.loc[data['PoolArea'] != 0, 'HavePool'] = 1
    data.loc[data['PoolArea'] == 0, 'HavePool'] = 0
    COLUMNS_TO_DROP_AT_END.extend(['PoolArea', 'PoolQC'])
    return data


def binarize_second_floor(data: pd.DataFrame):
    data[Has2ndFloor] = data[SecondFlrSF].apply(lambda x: 1 if x > 0 else 0)
    COLUMNS_TO_DROP_AT_END.extend([SecondFlrSF])
    return data


def binarize_garage(data: pd.DataFrame):
    data[HasGarage] = data[GarageArea].apply(lambda x: 1 if x > 0 else 0)
    COLUMNS_TO_DROP_AT_END.extend([GarageArea])
    return data


def binarize_basement(data: pd.DataFrame):
    data[HasBasement] = data[TotalBsmtSF].apply(lambda x: 1 if x > 0 else 0)
    COLUMNS_TO_DROP_AT_END.extend([TotalBsmtSF])
    return data


def binarize_fireplace(data: pd.DataFrame):
    data[HasFireplace] = data[Fireplaces].apply(lambda x: 1 if x > 0 else 0)
    COLUMNS_TO_DROP_AT_END.extend([Fireplaces])
    return data


def join_bathrooms(x: pd.DataFrame):
    x[TotalBathrooms] = x[FullBath] + 0.5 * x[HalfBath] + x[BsmtFullBath] + 0.5 * x[BsmtHalfBath]

    # columns that should probably be dropped:
    columns_to_drop = [FullBath, HalfBath, BsmtFullBath, BsmtHalfBath]

    COLUMNS_TO_DROP_AT_END.extend(columns_to_drop)

    return x


def group_exterior_covering(x: pd.DataFrame):
    """
    Exterior covering ('Exterior1st', 'Exterior2nd') has 2 features with 14 and 15 categories repectively
     - many of which are sparse.
    This function maps the 15 categories into 6 new categories.
    """

    def map_material(material):
        if material in ['Wd Sdng', 'Wd Shng', 'WdShing', 'AsphShn', 'Plywood']:  # Add or remove as per your dataset
            return 'Wood'
        elif material == 'VinylSd':
            return 'Vinyl'
        elif material in ['BrkComm', 'BrkFace', 'Brk Cmn']:
            return 'Brick'
        elif material in ['MetalSd', 'Stucco', 'CmentBd', 'CemntBd',
                          'ImStucc']:  # Adjust based on your dataset's values
            return 'Metal/Stucco/Cement'
        elif material in ['Stone', 'Other']:
            return material
        else:
            return 'Other'

    x[Exterior1stGroup] = x[Exterior1st].apply(map_material)
    x[Exterior2ndGroup] = x[Exterior2nd].apply(map_material)

    COLUMNS_TO_DROP_AT_END.extend([Exterior1st, Exterior2nd])

    return x


def group_roofstyle_roofmatl(data: pd.DataFrame) -> pd.DataFrame:
    """
    Joins rare categories to a single one
    """
    data[RoofStyleGroup] = data[RoofStyle].apply(
        lambda x: x if x in ['Gable', 'Hip'] else 'Other')  # Converts into 3 groups
    data[RoofMatlGroup] = data[RoofMatl].apply(
        lambda x: 'CompShg' if x == 'CompShg' else 'Other')  # Converts into 2 groups

    COLUMNS_TO_DROP_AT_END.extend([RoofStyle, RoofMatl])
    return data


def extract_asset_age(data: pd.DataFrame):
    """
    Calculate the asset age: YrSold - YearBuilt
    """
    data[AssetAge] = data[YrSold] - data[YearBuilt]

    COLUMNS_TO_DROP_AT_END.extend([YearBuilt])
    return data


def extract_garage_age(data: pd.DataFrame):
    """
    Calculate the asset age: YrSold - YearBuilt
    """
    data[GarageAge] = data[YrSold] - data[GarageYrBlt]

    COLUMNS_TO_DROP_AT_END.extend([GarageYrBlt])
    return data


def binarize_year_remodeled(data: pd.DataFrame) -> pd.DataFrame:
    """
    Binarize the YearRemodAdd to True/False
    """
    data[Remodeled] = (data[YearRemodAdd] - data[YearBuilt]).astype(bool).astype(int)
    COLUMNS_TO_DROP_AT_END.extend([YearRemodAdd])
    return data


class RelativeFeatureExtractor(FeatureExtractor):
    """
    Class for adding new features that are base on the relative position of the old ones.

    New features:
    - SizeRelativeToMedian - TotalArea divided by the median TotalArea of the neighborhood
    - IsInTopQuartile - 1 if TotalArea is in the top 15% of the neighborhood
    - NeibLevel - Neighborhood pricing level
    """

    def __init__(self, relative_LivArea=True,
                 sale_price_by_neighborhood=True):
        super().__init__()
        self._neighborhood_area_quantiles = None
        self._neighborhood_price_medians = None
        self._neigh_price_prct_25 = None
        self._neigh_price_prct_75 = None

        self._calc_relative_LivArea = relative_LivArea
        self._calc_sale_price_by_neighborhood = sale_price_by_neighborhood

    def fit(self, X: pd.DataFrame, y=None):
        if self._calc_relative_LivArea:
            self._fit_relative_liv_area(X, y)

        if self._calc_sale_price_by_neighborhood:
            self._fit_sale_price_by_neib(X, y)

        return super().fit(X, y)

    def _fit_relative_liv_area(self, X: pd.DataFrame, y=None):
        self._neighborhood_area_quantiles = X.groupby(Neighborhood)[TotalArea].agg(
            ['median', lambda x: np.quantile(x, 0.75)])
        self._neighborhood_area_quantiles.columns = ['50%', '75%']
        self._neighborhood_area_quantiles.reset_index(inplace=True)

    def _fit_sale_price_by_neib(self, X: pd.DataFrame, y=None):
        self._neighborhood_price_medians = X.groupby(Neighborhood)[SalePrice].median().reset_index(
            name='MedianSalePrice')

        percentile_25 = self._neighborhood_price_medians['MedianSalePrice'].quantile(0.25)
        percentile_75 = self._neighborhood_price_medians['MedianSalePrice'].quantile(0.75)

        def classify_neighborhood(price):
            if price <= percentile_25:
                return 0
            elif price >= percentile_75:
                return 1
            else:
                return 2

        self._neighborhood_price_medians[NeibLevel] = self._neighborhood_price_medians['MedianSalePrice'].apply(
            classify_neighborhood)

    def transform(self, X: pd.DataFrame):
        new_X = X.copy()

        if self._calc_relative_LivArea:
            X_with_stats = pd.merge(X, self._neighborhood_area_quantiles, on=Neighborhood, how='left').set_index(
                X.index)
            new_X[SizeRelativeToMedian] = X_with_stats[TotalArea] / X_with_stats['50%']
            new_X[IsInTopQuartile] = (X_with_stats[TotalArea] > X_with_stats['75%']).astype(int)

        if self._calc_sale_price_by_neighborhood:
            new_X = pd.merge(new_X, self._neighborhood_price_medians[[Neighborhood, NeibLevel]],
                             on=Neighborhood, how='left').set_index(X.index)

        return new_X


class CorrelatedNumericFeaturesDropper(FeatureExtractor):
    """
    Calculates the correlation between all pairs of numeric features in the dataframe, looks at all the pairs
    with correlation above the threshold, and drops the second feature in these pairs.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._columns_to_drop = []
        super().__init__()

    def fit(self, x: pd.DataFrame, y=None):
        logging.debug(f"Calculating correlation between numeric features in train set...")

        numeric_correlations = calc_numeric_feature_correlation(x)

        highly_correlated_numeric_features = [t for t in numeric_correlations if t[2] >= self.threshold]

        self._columns_to_drop = [t[1] for t in highly_correlated_numeric_features]

        logging.debug(f"Correlation calc done.")

        return self

    def transform(self, x):
        logging.debug(f"Dropping the following highly correlated numeric features: {self._columns_to_drop}")
        COLUMNS_TO_DROP_AT_END.extend(self._columns_to_drop)
        return x
