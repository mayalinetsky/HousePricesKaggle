"""
All parameters to use in model flow (preprocessing, model type, model param etc)


Note: "V0" refers to baseline

Note for teammates: When adding a pack to one of the 'manipulation packs' -
feature_extraction_packs/feature_target_separation_packs/preprocessing_packs/labeling_packs
make sure you add the key to all other manipulation packs.
Meaning you cannot add "V1" only to preprocessing_packs, without also adding 'V1' to all other manipulation packs.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from feature_extraction import FeatureExtractor
from feature_target_separation import separate_features_and_target
from labeling import produce_target
from preprocessing import baseline_preprocess
from preprocessors import NoFitPreProcessor, preprocess
from raw_data import get_raw_data
from raw_data_folding import BaseTrainValTestSplitter

get_raw_data_packs = {
    "V0": {"function": get_raw_data}
}

cross_validation_packs = {
    "TrainTrainTest": {"class": BaseTrainValTestSplitter,
                       "args": {}
                       }
}

# from name to arguments for a pipeline
feature_extraction_packs = {
    "V0": {"steps": [FeatureExtractor()]}
}

feature_target_separation_packs = {
    "V0": {"function": separate_features_and_target}
}

# from name to arguments for a pipeline
preprocessing_packs = {
    "V0": {"steps": [NoFitPreProcessor([baseline_preprocess]),
                     SimpleImputer(missing_values=pd.NA, strategy='mean').set_output(transform='pandas')]}
}

labeling_packs = {
    "V0": {"function": produce_target}
}

# from model name to params
model_grid_search_params = {
    "LinearRegression": {"class": LinearRegression,
                         "args": {}  # insert here list of hyper-parameters
                         }
}
