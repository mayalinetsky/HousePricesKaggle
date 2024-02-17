"""
Classes responsible to manipulate 'X features' into new features that are ready for model fitting.
These classes will be passed as a part of "steps" parameter is a sklearn pipeline.

Example:
    Normalization
    Binarization
    Value imputation
"""
import logging
import warnings
from typing import *
import pandas as pd
from scipy.optimize._optimize import BracketError
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox_normmax, ConstantInputWarning
from sklearn.base import TransformerMixin


class NoFitPreProcessor(TransformerMixin):
    """
    Preprocessors that applies processing function that require no fitting.

    Examples:
        - Remove all rows with nan values
        - Delete certain columns
        - Fill all nans with 0
    """

    def __init__(self, no_fit_functions: List[Callable[[pd.DataFrame], pd.DataFrame]], *args, **kwargs):
        """

        no_fit_functions: list of functions that manipulate 'X features' (not in place) and require no fitting.
        """
        super().__init__(*args, **kwargs)
        self._no_fit_functions = no_fit_functions

    def _apply_functions(self, X: pd.DataFrame):
        processed_X = X
        for func in self._no_fit_functions:
            processed_X = func(processed_X)
        return processed_X

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return self._apply_functions(X)

    def set_output(self, *, transform=None):
        pass


class SkewedFeaturesNormalizer(TransformerMixin):
    """
    Credit: https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1?scriptVersionId=20214677&cellId=38
    """
    def __init__(self):
        super().__init__()
        self._very_skewed_features = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics = []
        for i in X.columns:
            if X[i].dtype in numeric_dtypes:
                numerics.append(i)

        features_skewness = X.loc[:, numerics].apply(lambda x: skew(x)).sort_values(ascending=False)

        high_skew = features_skewness[features_skewness > 0.5]
        self._very_skewed_features = high_skew.index

        return self

    def transform(self, X: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            warnings.simplefilter("error", category=RuntimeWarning)
            for col in self._very_skewed_features:
                try:
                    X.loc[:, col] = boxcox1p(X[col], boxcox_normmax(X[col] + 1))
                except BaseException as e:
                    logging.debug(f"Could not perform boxcox1p transform on: {col}. ({e})")
            return X

    def set_output(self, *, transform=None):
        pass
