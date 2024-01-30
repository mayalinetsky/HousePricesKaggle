"""
Utilities module containing helper functions for the project
"""

from typing import Literal, Union
import numpy as np
import pandas as pd


def load_house_prices_data(source: Union[Literal['train'], Literal['test'], Literal['all']]):
    """
    Load train and\or test data.
    For information on columns, refer to data_description.txt

    Train data includes 'SalePrice'.
    Test data does not include 'SalePrice'.
    When requesting 'all' data, all columns are returned, but rows from test set will have NaN in the 'SalePrice'.
    """
    if source not in ['train', 'test', 'all']:
        raise (ValueError(f'Unsupported source type: {source}'))

    if source == 'train':
        return _load_train_data()

    if source == 'test':
        return _load_test_data()

    if source == 'all':
        train_df = _load_train_data()
        test_df = _load_test_data()

        return pd.concat([train_df, test_df], axis='rows', ignore_index=True)


def _load_train_data():
    return pd.read_csv(r'train.csv')

def _load_test_data():
    return pd.read_csv(r'test.csv')