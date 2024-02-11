"""
Module encapsulating "get_raw_data" phase that happens before splitting to folds
"""
from typing import *
import pandas as pd

from utils import load_house_prices_data


def get_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    Returns:
        (train_raw_data, test_raw_data)
    """
    train_raw_data = load_house_prices_data('train')
    test_raw_data = load_house_prices_data('test')

    return train_raw_data, test_raw_data
