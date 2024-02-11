"""
Classes responsible to manipulate 'X features' into new features that are ready for model fitting.
These classes will be passed as a part of "steps" parameter is a sklearn pipeline.

Example:
    Normalization
    Binarization
    Value imputation
"""

from typing import *
from preprocessing import *
import pandas as pd
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
