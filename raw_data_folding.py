"""
Split raw data into folds
"""
from custom_types import RawFold
import numpy as np
import pandas as pd


class BaseTrainValTestSplitter:
    """
    Produces only 1 fold with the validation set being the train set.
    (This is used for the baseline)
    """

    def __init__(self, *args, **kwargs):
        pass

    # couldn't understand if you put twice train_rawdata by mistake or just as a placeholder,
    # so I wrote a new method downwards, using "val_rawdata" instead


    # def split(self, train_rawdata, test_rawdata) -> list[RawFold]:
    #     return [RawFold(train_rawdata, train_rawdata, test_rawdata)]

    def split(self, train_rawdata, test_rawdata) -> list[RawFold]:
        """
        Splitting the train data into train and val according to the shortest Euclidean distance from test
        Assuming the data was preprocessed: contains no nans and categorical features were encoded as numeric
        """

        def _calculate_euclidean_distances(train, test):
            # Reshape to enable broadcasting
            train_reshaped = train[:, np.newaxis, :]

            # Calculate Euclidean distances for each pair of rows between train and test
            distances = np.linalg.norm(train_reshaped - test, axis=2)

            return distances

        def _find_closest_items_indices(train, test, fraction=0.2):
            distances = _calculate_euclidean_distances(train, test)

            # Find the indices of the closest rows
            closest_row_indices = np.argmin(distances, axis=0)

            # Determine the number of rows to select based on the fraction
            num_rows_to_select = int(train.shape[0] * fraction)

            # Sort the indices by distance and select the top ones
            sorted_indices = np.argsort(distances, axis=0)
            selected_row_indices = sorted_indices[:num_rows_to_select]

            return selected_row_indices

        # Calculate the 20% closest rows from train to each row in test
        selected_items_indices = _find_closest_items_indices(train_rawdata, test_rawdata, fraction=0.2)

        # Split the DataFrame into train and validation using iloc
        val_rawdata = train_rawdata.iloc[selected_items_indices.flatten()]
        train_rawdata = train_rawdata.drop(val_rawdata.index)

        return [RawFold(train_rawdata, val_rawdata, test_rawdata)]
