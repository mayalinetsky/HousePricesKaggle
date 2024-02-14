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
    - SizeRelativeToMedian TODO explain
    - IsInTopQuartile TODO explain
    """
    def __init__(self):
        super().__init__()
        self._neighborhood_stats = None

    def fit(self, X: pd.DataFrame, y=None):
        self._neighborhood_stats = X.groupby(Neighborhood)[GrLivArea].agg(
            ['median', lambda x: np.quantile(x, 0.75)])
        self._neighborhood_stats.columns = ['50%', '75%']
        self._neighborhood_stats.reset_index(inplace=True)

        return super().fit(X, y)

    def transform(self, X: pd.DataFrame):
        new_X = X.copy()
        X_with_stats = pd.merge(X, self._neighborhood_stats, on=Neighborhood, how='left').set_index(X.index)
        new_X['SizeRelativeToMedian'] = X_with_stats[GrLivArea] / X_with_stats['50%']
        new_X['IsInTopQuartile'] = (X_with_stats[GrLivArea] > X_with_stats['75%']).astype(int)
        return new_X
