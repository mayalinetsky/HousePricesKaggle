"""
Run all permutations listed in config

Note: temporarily, runs only the baseline model
"""
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV

from custom_types import ProcessedFold, X_y
from sklearn.pipeline import make_pipeline

from evaluation import rmse_log_scorer
from flow_config import (get_raw_data_packs,
                         feature_extraction_packs,
                         feature_target_separation_packs,
                         preprocessing_packs,
                         labeling_packs,
                         cross_validation_packs,
                         model_grid_search_params)
from model_flow import prepare_submission_csv

if __name__ == "__main__":
    """
    Run baseline model flow
    """
    logging.basicConfig(level=logging.INFO)

    # get raw data
    logging.info(f"Loading data...")
    train_raw_data, test_raw_data = get_raw_data_packs['V0']['function']()

    # split into folds
    logging.info(f"Splitting raw data into folds...")
    cv_splitter_config = cross_validation_packs['TrainTrainTest']
    cv_splitter = cv_splitter_config['class'](*cv_splitter_config['args'])
    raw_folds = cv_splitter.split(train_raw_data, test_raw_data)

    # extract features from raw data, separate target, preprocess
    feature_extraction_config = feature_extraction_packs['V0']
    feature_target_separation_func = feature_target_separation_packs['V0']['function']
    preprocessing_config = preprocessing_packs['V0']
    labeling_func = labeling_packs['V0']['function']

    logging.info(f"Preparing datasets...")
    processed_folds = []
    for fold_index, raw_fold in enumerate(raw_folds, start=1):
        logging.info(f"\tFold {fold_index}/{len(raw_folds)}")
        # fit extraction
        feature_extraction_pipeline = make_pipeline(*feature_extraction_config['steps'])
        feature_extraction_pipeline.fit(raw_fold.train_raw_data)

        unprocessed_x_ys: list[X_y] = []
        for raw_data_set in [raw_fold.train_raw_data, raw_fold.val_raw_data, raw_fold.test_raw_data]:
            # apply extraction
            extracted_data = feature_extraction_pipeline.transform(raw_data_set)

            # separate x features and target features
            X, y = feature_target_separation_func(extracted_data)
            unprocessed_x_ys.append((X, y))

        # preprocess and label
        preprocess_pipe = make_pipeline(*preprocessing_config['steps'])
        preprocess_pipe.fit(unprocessed_x_ys[0][0])

        processed_x_ys: list[X_y] = []
        for unprocessed_x_y in unprocessed_x_ys:
            X, y = unprocessed_x_y
            # apply preprocessing
            processed_X = preprocess_pipe.transform(X)

            # produce label
            processed_y = labeling_func(y)
            processed_x_ys.append((processed_X, processed_y))

        processed_fold = ProcessedFold(*processed_x_ys)
        processed_folds.append(processed_fold)

    logging.info(f"All datasets ready.")
    model_grid_search_config = model_grid_search_params['LinearRegression']

    test_predictions_per_fold: list[pd.Series] = []

    # hyper-param tuning on Prepped Folds
    logging.info(f"Tuning hyperparameters for each dataset...")
    for fold_index, fold in enumerate(processed_folds, start=1):
        logging.info(f"\tDataset {fold_index}/{len(processed_folds)}")

        train_val_combined_X = pd.concat([fold.train_X_y[0], fold.val_X_y[0]])
        train_val_combined_y = pd.concat([fold.train_X_y[1], fold.val_X_y[1]])

        train_val_indices = [(np.arange(len(fold.train_X_y[0])),
                              np.arange(len(fold.train_X_y[0]), len(train_val_combined_X)))]

        model = model_grid_search_config['class']()
        param_grid = model_grid_search_config['args']

        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=train_val_indices, scoring=rmse_log_scorer)
        clf.fit(train_val_combined_X, train_val_combined_y)

        logging.info(f"\tFound best estimator. Best score: {clf.best_score_}")
        best_model = clf.best_estimator_

        test_predictions = best_model.predict(fold.test_X_y[0])

        test_predictions_series = pd.Series(test_predictions, index=fold.test_X_y[0].index)

        test_predictions_per_fold.append(test_predictions_series)

    logging.info(f"Done tuning hyper-params. Preparing final predictions...")
    final_test_predictions = pd.concat(test_predictions_per_fold)
    prepare_submission_csv(final_test_predictions.index, final_test_predictions.values)

    logging.info(f"Done.")
