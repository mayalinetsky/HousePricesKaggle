"""
Module to prepare the dataframe to model fitting
"""
import pandas as pd


def preprocess(data: pd.DataFrame):
    """
    Prepare the data before model fitting:
    1. Converting some nan values to str
    """
    return _convert_nan_to_str(data)


def _convert_nan_to_str(data: pd.DataFrame):
    data = data.copy()
    relevant_columns = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageType',
                        'GarageFinish', 'GarageQual', 'GarageCond'
                        ]

    for feature in relevant_columns:
        data[feature].fillna(value='No', inplace=True)

    return data

