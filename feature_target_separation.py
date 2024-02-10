"""
Module responsible for taking extracted features and separating the features from the targets.
"""
import pandas as pd
from constants import SalePrice


def separate_features_and_target(extracted_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate input into "X features" and "Y features".

    Parameters:
    ----------
    extracted_features: pd.DataFrame output from a feature extraction

    Returns:
    -------
    X_Y: tuple with the features and target values separated (x_features, target_features)
    """
    # this is an example
    try:
        sale_price_df = extracted_features[[SalePrice]]
        features_no_target = extracted_features.drop(columns=[SalePrice])
    except KeyError:
        sale_price_df = pd.DataFrame(data=[], columns=[SalePrice], index=extracted_features.index)
        features_no_target = extracted_features

    return features_no_target, sale_price_df
