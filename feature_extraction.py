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
    x.drop(columns=close_porches + open_porches, inplace=True)
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

    # columns that should probably be dropped:
    columns_to_drop = [TotalBsmtSF, GrLivArea, BsmtFinSF2, SecondFlrSF]

    return x.drop(columns=columns_to_drop)


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

    return x.drop(columns=[Exterior1st, Exterior2nd])


def group_roofstyle_roofmatl(data: pd.DataFrame) -> pd.DataFrame:
    """
    Joins rare categories to a single one
    """
    data[RoofStyleGroup] = data[RoofStyle].apply(
        lambda x: x if x in ['Gable', 'Hip'] else 'Other')  # Converts into 3 groups
    data[RoofMatlGroup] = data[RoofMatl].apply(
        lambda x: 'CompShg' if x == 'CompShg' else 'Other')  # Converts into 2 groups

    return data.drop(columns=[RoofStyle, RoofMatl])


def extract_asset_age(data: pd.DataFrame):
    """
    Calculate the asset age. If the asset was remodeled, the age is calculated using the remodel year.
    TODO this means we will treat a 50 year old appt that has just been remodeled as a new appt?
    """
    data[AssetAge] = data[YrSold] - data[[YearBuilt, YearRemodAdd]].max(axis=1)

    return data.drop(columns=[YearBuilt, YearRemodAdd])


class RelativeFeatureExtractor(FeatureExtractor):
    """
    Class for adding new features that are base on the relative position of the old ones.

    New features:
    - SizeRelativeToMedian - TotalArea divided by the median TotalArea of the neighborhood
    - IsInTopQuartile - 1 if TotalArea is in the top 15% of the neighborhood
    - NeibLevel - Neighborhood pricing level
    """

    def __init__(self, relative_LivArea=True, sale_price_by_neighborhood=True):
        super().__init__()
        self._neighborhood_area_quantiles = None
        self._neighborhood_price_medians = None
        self._neigh_price_prct_25 = None
        self._neigh_price_prct_75 = None

        self._relative_LivArea = relative_LivArea
        self._sale_price_by_neighborhood = sale_price_by_neighborhood

    def fit(self, X: pd.DataFrame, y=None):
        if self._relative_LivArea:
            self._neighborhood_area_quantiles = X.groupby(Neighborhood)[TotalArea].agg(
                ['median', lambda x: np.quantile(x, 0.75)])
            self._neighborhood_area_quantiles.columns = ['50%', '75%']
            self._neighborhood_area_quantiles.reset_index(inplace=True)

        if self._sale_price_by_neighborhood:
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

        return super().fit(X, y)

    def transform(self, X: pd.DataFrame):
        new_X = X.copy()

        if self._relative_LivArea:
            X_with_stats = pd.merge(X, self._neighborhood_area_quantiles, on=Neighborhood, how='left').set_index(
                X.index)
            new_X[SizeRelativeToMedian] = X_with_stats[TotalArea] / X_with_stats['50%']
            new_X[IsInTopQuartile] = (X_with_stats[TotalArea] > X_with_stats['75%']).astype(int)

        if self._sale_price_by_neighborhood:
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
        return x.drop(columns=self._columns_to_drop, errors='ignore')
