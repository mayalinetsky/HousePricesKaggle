"""
Module responsible for getting 'Y features' and converting them into target values for predictions.
"""
import numpy as np
import pandas as pd

from constants import SalePrice


def produce_target(target_features: pd.DataFrame) -> pd.Series:
    """
    convert input into target values for predictions
    """
    return target_features[SalePrice]


def produce_log_target(target_features: pd.DataFrame) -> pd.Series:
    """
    convert input into target values for predictions
    """
    return np.log1p(target_features[SalePrice].astype(float))
