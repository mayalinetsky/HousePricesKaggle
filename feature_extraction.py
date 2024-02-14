"""
Module encapsulating "feature extraction" phase that happens after splitting to folds.
It gets raw data and transforms it into features ready for preprocessing and labeling.
"""
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from constants import *


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

    x['TotClosePorchSF'] = x[close_porches].apply(join_areas, axis=1)
    x['TotOpenPorchSF'] = x[open_porches].apply(join_areas, axis=1)
    x.drop(columns=close_porches + open_porches, inplace=True)
    return x


class RelativeFeatureExtractor(FeatureExtractor):
    """
    Class for adding new features that are base on the relative position of the old ones.

    New features:
    - SizeRelativeToMedian - GrLivArea divided by the median GrLivArea of the neighborhood
    - IsInTopQuartile - 1 if GrLivArea is in the top 15% of the neighborhood
    """

    def __init__(self, relative_GrLivArea=True, sale_price_by_neighborhood=True):
        super().__init__()
        self._neighborhood_area_quantiles = None
        self._neighborhood_price_medians = None
        self._neigh_price_prct_25 = None
        self._neigh_price_prct_75 = None

        self._relative_GrLivArea = relative_GrLivArea
        self._sale_price_by_neighborhood = sale_price_by_neighborhood

    def fit(self, X: pd.DataFrame, y=None):
        if self._relative_GrLivArea:
            self._neighborhood_area_quantiles = X.groupby(Neighborhood)[GrLivArea].agg(
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
                    return 'Low'
                elif price >= percentile_75:
                    return 'High'
                else:
                    return 'Medium'

            self._neighborhood_price_medians['NeibLevel'] = self._neighborhood_price_medians['MedianSalePrice'].apply(
                classify_neighborhood)

        return super().fit(X, y)

    def transform(self, X: pd.DataFrame):
        new_X = X.copy()

        if self._relative_GrLivArea:
            X_with_stats = pd.merge(X, self._neighborhood_area_quantiles, on=Neighborhood, how='left').set_index(
                X.index)
            new_X['SizeRelativeToMedian'] = X_with_stats[GrLivArea] / X_with_stats['50%']
            new_X['IsInTopQuartile'] = (X_with_stats[GrLivArea] > X_with_stats['75%']).astype(int)

        if self._sale_price_by_neighborhood:
            new_X = pd.merge(new_X, self._neighborhood_price_medians[['Neighborhood', 'NeibLevel']],
                             on='Neighborhood', how='left').set_index(X.index)
        return new_X
