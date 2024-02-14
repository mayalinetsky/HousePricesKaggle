"""
Module encapsulating "get_raw_data" phase that happens before splitting to folds
"""
import logging
from typing import *
import pandas as pd

from constants import GrLivArea
from utils import load_house_prices_data


def get_raw_data(filter_samples: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param filter_samples: weather to filter samples from the training set or not
    Returns:
        (train_raw_data, test_raw_data)
    """
    train_raw_data = load_house_prices_data('train')

    if filter_samples:
        logging.debug(f"Filtering raw training data...")
        train_raw_data = train_raw_data[train_raw_data[GrLivArea] <= 4000]

    test_raw_data = load_house_prices_data('test')

    return train_raw_data, test_raw_data
