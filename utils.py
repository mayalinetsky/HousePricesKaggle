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


def calc_num_missing_vals_per_col(data: pd.DataFrame):
    """
    Return a pd.Series with index of a column name, and values of the number of missing values for the column.
    Does not include columns with no missing values.
    Sorted from smallest to biggest.
    """

    def count_nans(column):
        return column.isna().sum()

    num_of_nans = data.apply(count_nans)
    col_to_num_of_nans_display = num_of_nans[num_of_nans != 0].sort_values(ascending=False)
    return col_to_num_of_nans_display


def calc_numeric_feature_correlation(features_df: pd.DataFrame) -> list[tuple[str, str, float]]:
    """
    Calculate the pearson correlation between all pairs of numeric features in the dataframe.
    """
    correlation_matrix = features_df.corr(numeric_only=True)
    correlations = []

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            feature1 = correlation_matrix.index[i]
            feature2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]

            correlations.append((feature1, feature2, round(correlation, 3)))

    return correlations


def calc_categorical_feature_correlation_to_target(df: pd.DataFrame):
    """
    Calculate the pearson correlation between all categorical features in the dataframe to the sale price.
    This is done using one-hot encoding of the features.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    correlation = df_encoded.corr()
    categorical_correlation = correlation['SalePrice'].drop(df.select_dtypes(include=[np.number]).columns)
    sorted_correlation = categorical_correlation.sort_values(ascending=False)

    return sorted_correlation

