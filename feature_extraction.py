"""
Module encapsulating "feature extraction" phase that happens after splitting to folds.
It gets raw data and transforms it into features ready for preprocessing and labeling.
"""
from typing import *
from sklearn.base import TransformerMixin


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

