"""
Module encapsulating "feature extraction" phase that happens after splitting to folds.
It gets raw data and transforms it into features ready for preprocessing and labeling.
"""
from typing import *

import pandas as pd
from sklearn.base import TransformerMixin
from constants import *


class FeatureExtractor(TransformerMixin):
    """
    Base class for transforming raw data into new features.

    For example: one-hot encoding
    """
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, y=None):
        return x

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

