"""
Run all permutations listed in config.
See the main_flow.png for a diagram.

Note: temporarily, runs only the baseline model
"""
import pandas as pd
import logging
from flow_config import (get_raw_data_packs,
                         feature_extraction_packs,
                         feature_target_separation_packs,
                         preprocessing_packs,
                         labeling_packs,
                         cross_validation_packs,
                         model_grid_search_params)
from model_flow import prepare_submission_csv, process_fold, tune_hyper_params, concat_folds_for_cv
from raw_data_folding import BaseTrainValTestSplitter

if __name__ == "__main__":
    """
    Run baseline model flow
    """
    logging.basicConfig(level=logging.DEBUG)

    GET_RAW_DATA_PACK = "V1"
    FEATURE_EXTRACTION_PACK = "V1"
    FEAT_TARGET_SEPARATION_PACK = "V0"
    PREPROCESSING_PACK = "V8"
    LABELING_PACK = "V1"
    CROSS_VALIDATION_PACK = "KFold"
    MODEL_PACK = 'GradientBoostingRegressor'

    # get raw data
    logging.info(f"Loading data...")
    train_raw_data, test_raw_data = get_raw_data_packs[GET_RAW_DATA_PACK]['function']()

    # split into folds
    logging.info(f"Preparing dataset...")
    cv_splitter_config = cross_validation_packs[CROSS_VALIDATION_PACK]
    cv_splitter = cv_splitter_config['class'](**cv_splitter_config['args'])

    # extract features from raw data, separate target, preprocess
    processed_folds = []
    for fold_index, raw_fold in enumerate(BaseTrainValTestSplitter().split(train_raw_data, test_raw_data), start=1):
        # logging.info(f"\tFold {fold_index}/{len(raw_folds)}")
        processed_fold = process_fold(raw_fold,
                                      feature_extraction_pack=feature_extraction_packs[FEATURE_EXTRACTION_PACK],
                                      feature_target_separation_pack=feature_target_separation_packs[FEAT_TARGET_SEPARATION_PACK],
                                      preprocessing_pack=preprocessing_packs[PREPROCESSING_PACK],
                                      labeling_pack=labeling_packs[LABELING_PACK])
        processed_folds.append(processed_fold)

    logging.info(f"All folds ready.")
    model_grid_search_config = model_grid_search_params[MODEL_PACK]

    # logging.info(f"Combining folds into a single input for GridSearchCV")
    # train_val_combined_X, train_val_combined_y, train_val_indices, test_combined_X = concat_folds_for_cv(processed_folds)

    # hyper-param tuning on Prepped Folds
    logging.info(f"Tuning hyperparameters to optimize average score over all folds...")
    the_only_fold = processed_folds[0]
    clf = tune_hyper_params(the_only_fold.train_X_y[0],
                            the_only_fold.train_X_y[1],
                            cv_splitter,
                            model_grid_search_config, scorer=labeling_packs[LABELING_PACK]['scorer'])

    logging.info(f"Found best estimator. Best cv score: {clf.best_score_}, Best params: {clf.best_params_}")

    best_model = clf.best_estimator_

    logging.info(f"\nPredicting on test set...")
    test_predictions_ = best_model.predict(the_only_fold.test_X_y[0])

    test_predictions = labeling_packs[LABELING_PACK]['inverse'](test_predictions_)

    test_predictions_series = pd.Series(test_predictions, index=the_only_fold.test_X_y[0].index)

    logging.info(f"Preparing final predictions...")
    prepare_submission_csv(test_predictions_series.index,
                           test_predictions_series.values,
                           clf.best_score_,
                           GET_RAW_DATA_PACK,
                           CROSS_VALIDATION_PACK,
                           FEATURE_EXTRACTION_PACK,
                           FEAT_TARGET_SEPARATION_PACK,
                           PREPROCESSING_PACK,
                           LABELING_PACK,
                           MODEL_PACK)

    logging.info(f"Done.")
