import pandas as pd
import numpy as np

from flow_config import get_raw_data_packs
# Imports
from utils import load_house_prices_data
from utils import calc_num_missing_vals_per_col, np
from plot_utils import plot_num_missing_values
from plot_utils import plot_price_dist_per_year
from plot_utils import plot_column_histograms
from plot_utils import plot_numeric_features_correlation_to_target
from plot_utils import plot_head_and_tail_categorical_corr_to_target
from utils import calc_categorical_feature_correlation_to_target
from utils import calc_numeric_feature_correlation
from plot_utils import plot_mean_price_and_stddev_per_category
from preprocessing import preprocess
from plot_utils import plot_number_of_sales_and_prices_across_time

if __name__ == "__main__":
    train_raw_data, test_raw_data = get_raw_data_packs['V0']['function']()
    print(train_raw_data)