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
from model_flow import prepare_submission_csv, process_fold, tune_hyper_params

if __name__ == "__main__":
    """
    Run baseline model flow
    """
    logging.basicConfig(level=logging.DEBUG)

    # get raw data
    logging.info(f"Loading data...")
    train_raw_data, test_raw_data = get_raw_data_packs['V1']['function']()

    # split into folds
    logging.info(f"Splitting raw data into folds...")
    cv_splitter_config = cross_validation_packs['TrainTrainTest']
    cv_splitter = cv_splitter_config['class'](*cv_splitter_config['args'])
    raw_folds = cv_splitter.split(train_raw_data, test_raw_data)

    # extract features from raw data, separate target, preprocess
    logging.info(f"Preparing datasets...")
    processed_folds = []
    for fold_index, raw_fold in enumerate(raw_folds, start=1):
        logging.info(f"\tFold {fold_index}/{len(raw_folds)}")
        processed_fold = process_fold(raw_fold,
                                      feature_extraction_pack=feature_extraction_packs["V1"],
                                      feature_target_separation_pack=feature_target_separation_packs["V0"],
                                      preprocessing_pack=preprocessing_packs["V1"],
                                      labeling_pack=labeling_packs["V0"])
        processed_folds.append(processed_fold)

    logging.info(f"All datasets ready.")
    names = ['LinearRegression', 'RidgeClassifier', 'LogisticRegression', 'SGDClassifier', 'GaussianNB',
             'MultinomialNB', 'KNeighborsClassifier', 'NearestCentroid', 'RandomForestClassifier', 'LinearSVC', 'SVC']
    average_results = list()

    for name_index, name in enumerate(names):
        model_grid_search_config = model_grid_search_params[name]

        test_pred_per_fold: list[pd.Series] = []

        # hyper-param tuning on Prepped Folds
        logging.info(f"{name_index}. {name}:Tuning hyperparameters for each dataset...")
        model_average_score = 0

        for fold_index, fold in enumerate(processed_folds, start=1):
            logging.info(f"\tDataset {fold_index}/{len(processed_folds)}")

            clf = tune_hyper_params(fold, model_grid_search_config)

            logging.info(f"\tFound best estimator for {name}. Best score: {clf.best_score_}")
            best_model = clf.best_estimator_

            test_pred = best_model.predict(fold.test_X_y[0])

            test_pred_series = pd.Series(test_pred, index=fold.test_X_y[0].index)

            test_pred_per_fold.append(test_pred_series)

            logging.info(f"Done tuning hyper-params. Preparing final predictions...")
            final_test_pred = pd.concat(test_pred_per_fold)

        prepare_submission_csv(final_test_pred.index, final_test_pred.values)

        logging.info(f"Done.")
