"""
Helper functions for main flow.
See the main_flow.png for a diagram.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from evaluation import rmse_log_scorer
from custom_types import RawFold, ProcessedFold, X_y


def prepare_submission_csv(ids: np.ndarray,
                           sale_price_prediction: np.ndarray,
                           weighted_val_score: float,
                           raw_data_pack: str,
                           cross_val_pack: str,
                           feat_ext_pack: str,
                           feat_target_pack: str,
                           preproc_pack: str,
                           labeling_pack: str,
                           model_pack: str):
    int_ids = ids.astype(int)
    submission_df = pd.DataFrame(data={"Id": int_ids, "SalePrice": sale_price_prediction})

    datetime_str = time.strftime('%Y-%m-%d--%H-%M-%S')
    config_str = f"raw{raw_data_pack}_cv{cross_val_pack}_fe{feat_ext_pack}_ft{feat_target_pack}_pr{preproc_pack}_la{labeling_pack}_mo{model_pack}"

    submission_df.to_csv(f"predictions_{weighted_val_score:.3f}_{config_str}_{datetime_str}.csv",
                         index=False)


def process_fold(raw_fold: RawFold,
                 feature_extraction_pack: dict,
                 feature_target_separation_pack: dict,
                 preprocessing_pack: dict,
                 labeling_pack: dict) -> ProcessedFold:
    """
    Apply feature extraction, then separate X features and y features,
    then preprocess X features and produce labels from y features.
    """
    # fit feature extractor
    feature_extraction_pipeline = make_pipeline(*feature_extraction_pack['steps'])
    feature_extraction_pipeline.fit(raw_fold.train_raw_data)

    unprocessed_x_ys: list[X_y] = []
    for raw_data_set in [raw_fold.train_raw_data, raw_fold.val_raw_data, raw_fold.test_raw_data]:
        # apply extraction
        extracted_data = feature_extraction_pipeline.transform(raw_data_set)

        # separate x features and target features
        X, y = feature_target_separation_pack['function'](extracted_data)
        unprocessed_x_ys.append((X, y))

    # preprocess and label
    preprocess_pipe = make_pipeline(*preprocessing_pack['steps']).set_output(transform="pandas")
    preprocess_pipe.fit(unprocessed_x_ys[0][0])

    processed_x_ys: list[X_y] = []
    for unprocessed_x_y in unprocessed_x_ys:
        X, y = unprocessed_x_y
        # apply preprocessing
        processed_X = preprocess_pipe.transform(X)

        # produce label
        processed_y = labeling_pack['function'](y)
        processed_x_ys.append((processed_X, processed_y))

    return ProcessedFold(*processed_x_ys)


def tune_hyper_params(train_val_combined_X: pd.DataFrame,
                      train_val_combined_y: pd.Series,
                      cv_indices: list[tuple[np.ndarray, np.ndarray]], model_pack: dict, scorer):

    model = model_pack['class']()
    param_grid = model_pack['args']

    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_indices, scoring=scorer)
    clf.fit(train_val_combined_X, train_val_combined_y)

    return clf


def concat_folds_for_cv(processed_folds: list[ProcessedFold]):
    """
    Return a dataframe that is a concatenation of all folds' train and validation sets,
    and an iterator that yields (train indices, val indices) tuples for each folds (this is used in gridsearch)
    """
    train_val_combined_X = pd.DataFrame()
    train_val_combined_y = pd.Series()
    train_val_indices = []
    test_combined_X = pd.DataFrame()

    for index, processed_fold in enumerate(processed_folds, start=1):
        train_X, train_y = processed_fold.train_X_y
        val_X, val_y = processed_fold.val_X_y
        test_X, _ = processed_fold.test_X_y

        index_shift = len(train_val_combined_X)
        train_indices = np.arange(index_shift, index_shift+len(train_X))
        val_indices = np.arange(index_shift+len(train_X), index_shift+len(train_X)+len(val_X))

        train_val_indices.append((train_indices, val_indices))

        train_val_combined_X = pd.concat([train_val_combined_X,
                                          train_X,
                                          val_X])

        train_val_combined_y = pd.concat([train_val_combined_y,
                                          train_y,
                                          val_y])

        test_combined_X = pd.concat([test_combined_X, test_X], ignore_index=False)

    # because each fold can have different features,
    # when concatenating all folds we can get nan values in features that are not present in all folds
    train_val_combined_X.fillna(0, inplace=True)
    test_combined_X.fillna(0, inplace=True)
    return train_val_combined_X, train_val_combined_y, train_val_indices, test_combined_X
