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


def prepare_submission_csv(ids: np.ndarray, sale_price_prediction: np.ndarray, title: str = 'single model'):
    int_ids = ids.astype(int)
    submission_df = pd.DataFrame(data={"Id": int_ids, "SalePrice": sale_price_prediction})

    datetime_str = time.strftime('%Y-%m-%d--%H-%M-%S')
    submission_df.to_csv(f"results/{title}_predictions_{datetime_str}.csv",
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


def tune_hyper_params(processed_fold: ProcessedFold, model_pack: dict):
    train_val_combined_X = pd.concat([processed_fold.train_X_y[0], processed_fold.val_X_y[0]])
    train_val_combined_y = pd.concat([processed_fold.train_X_y[1], processed_fold.val_X_y[1]])

    train_val_indices = [(np.arange(len(processed_fold.train_X_y[0])),
                          np.arange(len(processed_fold.train_X_y[0]), len(train_val_combined_X)))]

    model = model_pack['class']()
    param_grid = model_pack['args']

    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=train_val_indices, scoring=rmse_log_scorer)
    clf.fit(train_val_combined_X, train_val_combined_y)

    return clf