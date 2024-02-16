"""
All parameters to use in model flow (preprocessing, model type, model param etc)


Note: "V0" refers to baseline

Note for teammates: When adding a pack to one of the 'manipulation packs' -
feature_extraction_packs/feature_target_separation_packs/preprocessing_packs/labeling_packs
make sure you add the key to all other manipulation packs.
Meaning you cannot add "V1" only to preprocessing_packs, without also adding 'V1' to all other manipulation packs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from constants import *
from evaluation import rmse_log_scorer, rmse_scorer
from feature_extraction import FeatureExtractor, join_porch_areas, RelativeFeatureExtractor, join_liv_bsmt_areas, \
    CorrelatedNumericFeaturesDropper, group_exterior_covering, group_roofstyle_roofmatl, extract_asset_age, \
    binarize_year_remodeled, join_bathrooms, binarize_pool, binarize_second_floor, binarize_garage, binarize_basement, \
    binarize_fireplace
from feature_target_separation import separate_features_and_target
from labeling import produce_target, produce_log_target
from preprocessing import baseline_preprocess, drop_known_columns, preprocess, drop_globally_gathered_columns
from preprocessors import NoFitPreProcessor
from raw_data import get_raw_data
from raw_data_folding import BaseTrainValTestSplitter, AllUntilMonthSplitter

get_raw_data_packs = {
    "V0": {"function": get_raw_data},
    "V1": {"function": lambda: get_raw_data(filter_samples=True)}
}

cross_validation_packs = {
    "TrainTrainTest": {"class": BaseTrainValTestSplitter,
                       "args": {}
                       },
    "AllUntilMonthSplitter": {"class": AllUntilMonthSplitter,
                              "args": {}}
}

# from name to arguments for a pipeline
feature_extraction_packs = {
    "V0": {"steps": [FeatureExtractor()]},
    "V1": {"steps": [NoFitPreProcessor([join_porch_areas,
                                        join_liv_bsmt_areas,
                                        group_exterior_covering,
                                        group_roofstyle_roofmatl,
                                        binarize_year_remodeled,
                                        extract_asset_age,
                                        join_bathrooms,
                                        binarize_pool,
                                        binarize_second_floor,
                                        binarize_garage,
                                        binarize_basement,
                                        binarize_fireplace]),
                     RelativeFeatureExtractor()]}
}

feature_target_separation_packs = {
    "V0": {"function": separate_features_and_target}
}

COMMON_CATEGORICAL_FEATURES1 = [ExterQual, ExterCond, HeatingQC, KitchenQual]
COMMON_CATEGORICAL_ORDINAL_ENCODER1 = OrdinalEncoder(
    categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']] * len(COMMON_CATEGORICAL_FEATURES1),
    handle_unknown='use_encoded_value',
    unknown_value=-1)

COMMON_CATEGORICAL_FEATURES2 = [BsmtQual, BsmtCond, FireplaceQu, GarageQual, GarageCond]
COMMON_CATEGORICAL_ORDINAL_ENCODER2 = OrdinalEncoder(
    categories=[['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']] * len(COMMON_CATEGORICAL_FEATURES2),
    handle_unknown='use_encoded_value',
    unknown_value=-1)

UNCOMMON_CATEGORICAL_FEATURES = [LotShape, Utilities, LandSlope, BsmtExposure, Functional,
                                 GarageFinish]
UNCOMMON_CATEGORICAL_ORDINAL_ENCODER = OrdinalEncoder(categories=[['IR3', 'IR2', 'IR1', 'Reg'],  # LotShape
                                                                  ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],  # Utilities
                                                                  ['Sev', 'Mod', 'Gtl'],  # LandSlope
                                                                  ['None', 'No', 'Mn', 'Av', 'Gd'],  # BsmtExposure
                                                                  ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2',
                                                                   'Min1',
                                                                   'Typ'],  # Functional
                                                                  ['None', 'No', 'Unf', 'RFn', 'Fin']
                                                                  # GarageFinish # 'None' was NA before handling
                                                                  # missing values
                                                                  ],
                                                      handle_unknown='use_encoded_value',
                                                      unknown_value=-1)
# from name to arguments for a pipeline
preprocessing_packs = {
    "V0": {"steps": [NoFitPreProcessor([baseline_preprocess]),
                     SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform='pandas'),
                     RobustScaler()]},
    "V1": {"steps": [NoFitPreProcessor([preprocess]),
                     ColumnTransformer(transformers=[
                         ('common_cat', COMMON_CATEGORICAL_ORDINAL_ENCODER1, COMMON_CATEGORICAL_FEATURES1),
                         ('common_cat2', COMMON_CATEGORICAL_ORDINAL_ENCODER2, COMMON_CATEGORICAL_FEATURES2),
                         ('uncommon_cat', UNCOMMON_CATEGORICAL_ORDINAL_ENCODER, UNCOMMON_CATEGORICAL_FEATURES)
                     ],
                         remainder='passthrough'
                     ),
                     NoFitPreProcessor([drop_known_columns, drop_globally_gathered_columns]),
                     OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False),
                     SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform='pandas'),
                     RobustScaler()
                     ],
           },
    "V2": {"steps": [NoFitPreProcessor([baseline_preprocess, drop_globally_gathered_columns]),
                     SimpleImputer(missing_values=pd.NA, strategy="mean").set_output(transform='pandas'),
                     RobustScaler()
                     ],
           },
    "V3": {"steps": [NoFitPreProcessor([preprocess,
                                        drop_globally_gathered_columns,
                                        lambda d: d.select_dtypes(include='number')]),
                     SimpleImputer(missing_values=pd.NA, strategy="mean").set_output(transform='pandas'),
                     RobustScaler()
                     ],
           },
    "V4": {"steps": [NoFitPreProcessor([preprocess,
                                        drop_known_columns,
                                        drop_globally_gathered_columns,
                                        lambda d: d.select_dtypes(include='number')]),
                     SimpleImputer(missing_values=pd.NA, strategy="mean").set_output(transform='pandas'),
                     RobustScaler()
                     ],
           },
    "V5": {"steps": [NoFitPreProcessor([preprocess,
                                        drop_known_columns,
                                        drop_globally_gathered_columns,
                                        lambda d: d.select_dtypes(include='number')]),
                     SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=0).set_output(
                         transform='pandas'),
                     RobustScaler()
                     ],
           },
    "V6": {"steps": [NoFitPreProcessor([preprocess]),
                     ColumnTransformer(transformers=[
                         ('common_cat', COMMON_CATEGORICAL_ORDINAL_ENCODER1, COMMON_CATEGORICAL_FEATURES1),
                         ('common_cat2', COMMON_CATEGORICAL_ORDINAL_ENCODER2, COMMON_CATEGORICAL_FEATURES2),
                         ('uncommon_cat', UNCOMMON_CATEGORICAL_ORDINAL_ENCODER, UNCOMMON_CATEGORICAL_FEATURES),
                     ],
                         remainder='passthrough',
                         verbose_feature_names_out=False,
                     ),
                     NoFitPreProcessor([drop_known_columns,
                                        drop_globally_gathered_columns,
                                        lambda d: d.select_dtypes(include='number')]),
                     SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value=0).set_output(
                         transform='pandas'),
                     RobustScaler()
                     ],
           }

}

labeling_packs = {
    "V0": {"function": produce_target,
           "scorer": rmse_log_scorer,
           "inverse": lambda predictions: predictions},
    "V1": {"function": produce_log_target,
           "scorer": rmse_scorer,
           "inverse": lambda predictions: np.expm1(predictions)}
}

# from model name to params
model_grid_search_params = {
    "LinearRegression": {"class": LinearRegression,
                         "args": {}  # insert here list of hyper-parameters
                         },
    "Ridge": {"class": Ridge,
              "args": {'alpha': np.logspace(-4, 1, 10),
                       'solver': ['auto']}
              },
    "RandomForestRegressor": {"class": RandomForestRegressor,
                              "args": {'n_jobs': [-1],
                                       # 'n_estimators': [50, 100, 1000, 3000],
                                       # 'max_depth': [4],
                                       # 'max_features': ['sqrt', 1.0],
                                       # 'min_samples_leaf': [15],
                                       # 'criterion': ['squared_error', 'absolute_error'],
                                       # 'oob_score': [True],
                                       # 'ccp_alpha': [0, 0.01, 0.1, 1, 10, 100],
                                       'random_state': [42]
                                       }
                              },
    "TunedRandomForestRegressor": {"class": RandomForestRegressor,
                                   "args": {'n_jobs': [-1],
                                            'n_estimators': [50, 100, 1000, 3000],
                                            'max_depth': [None, 4, 10, 20, 100],
                                            'max_features': ['sqrt', 1.0],
                                            'min_samples_leaf': [1, 2, 3, 10, 15],
                                            'criterion': ['squared_error', 'absolute_error'],
                                            'oob_score': [False, True],
                                            'ccp_alpha': [0, 0.01, 0.1, 1, 10, 100],
                                            'random_state': [42]
                                            }
                                   },
    "GradientBoostingRegressor": {"class": GradientBoostingRegressor,
                                  "args": {'n_estimators': [100, 1000, 3000],
                                           'learning_rate': np.logspace(-4, 1, 10),
                                           'max_depth': [4],
                                           'max_features': ['sqrt'],
                                           'min_samples_leaf': [15],
                                           'loss': ['huber'],
                                           'n_iter_no_change': [5],
                                           'ccp_alpha': [0, 0.01, 0.1, 1, 10, 100],
                                           'random_state': [42]}
                                  }
}
