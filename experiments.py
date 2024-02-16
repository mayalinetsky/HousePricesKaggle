import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

train_origin = pd.read_csv('train.csv')
train_for_play = train_origin.copy()
train_for_play.head()

train_for_play[train_for_play['YearRemodAdd'] < 1960]['YearRemodAdd'].value_counts().sort_index()

train_for_play[(train_for_play['YearBuilt'] < 1960) & (train_for_play['YearBuilt'] > 1949)]['YearBuilt'].value_counts().sort_index()

train_for_play[train_for_play['YearRemodAdd'] == 1950][['YearRemodAdd', 'YearBuilt']].value_counts().sort_index()

train_for_play['age_when_sold'] = train_for_play['YrSold'] - train_for_play['YearBuilt']
train_for_play['remodeling_age_when_sold'] = train_for_play['YrSold'] - train_for_play['YearRemodAdd']

train_for_play['age_when_sold'].value_counts()

fig, ax = plt.subplots(1, 1)
ax.hist(train_for_play['age_when_sold'], bins=20)

fig, ax = plt.subplots(1, 1)
ax.hist(train_for_play['remodeling_age_when_sold'])

fig, ax = plt.subplots(1, 1)
ax.scatter(train_for_play['age_when_sold'], train_for_play['SalePrice'])
ax.set_title('sale price vs. age of dwelling')
ax.set_xlabel('age when sold')
ax.set_ylabel('price')

fig, ax = plt.subplots(1, 1)
ax.scatter(train_for_play['remodeling_age_when_sold'], train_for_play['SalePrice'])
ax.set_title('sale price vs. remodeling age of dwelling')
ax.set_xlabel('remodeling age when sold')
ax.set_ylabel('price')

train_for_play.loc[train_for_play['YearRemodAdd'] == train_for_play['YearBuilt']]['SalePrice'].mean()

remodeled_same_as_built = train_for_play.loc[train_for_play['YearRemodAdd'] == train_for_play['YearBuilt']]
remodeled_not_same_as_built = train_for_play.loc[train_for_play['YearRemodAdd'] != train_for_play['YearBuilt']]

# +
fig, ax = plt.subplots(1, 2, figsize=(16,5))
ax[0].scatter(remodeled_same_as_built['remodeling_age_when_sold'], remodeled_same_as_built['SalePrice'])
ax[0].set_title('remodeling year same as built')
ax[0].set_xlabel('remodeling age when sold')
ax[0].set_ylabel('price')

ax[1].scatter(remodeled_not_same_as_built['remodeling_age_when_sold'], remodeled_not_same_as_built['SalePrice'])
ax[1].set_title('remodeling year NOT same as built')
ax[1].set_xlabel('remodeling age when sold')
ax[1].set_ylabel('price')

# +
# Filter to get rows where 'YearRemodAdd' is 1950
filtered_rows = train_for_play['YearRemodAdd'] == 1950

# Calculate the mean of 'YearRemodAdd' and 'YearBuilt' for these rows
mean_years = np.mean([train_for_play.loc[filtered_rows, 'YearRemodAdd'], train_for_play.loc[filtered_rows, 'YearBuilt']], axis=0)

# Update 'YearRemodAdd' for rows where it is 1950
train_for_play.loc[filtered_rows, 'YearRemodAdd'] = mean_years

# -

train_for_play['remodeling_age_when_sold'] = train_for_play['YrSold'] - train_for_play['YearRemodAdd']

fig, ax = plt.subplots(1, 1)
ax.hist(train_for_play['remodeling_age_when_sold'])

fig, ax = plt.subplots(1, 1)
ax.scatter(train_for_play['remodeling_age_when_sold'], train_for_play['SalePrice'])
ax.set_title('sale price vs. remodeling age of dwelling')
ax.set_xlabel('remodeling age when sold')
ax.set_ylabel('price')

remodeled_same_as_built = train_for_play.loc[train_for_play['YearRemodAdd'] == train_for_play['YearBuilt']]
remodeled_not_same_as_built = train_for_play.loc[train_for_play['YearRemodAdd'] != train_for_play['YearBuilt']]

# +
fig, ax = plt.subplots(1, 2, figsize=(16,5))
ax[0].scatter(remodeled_same_as_built['remodeling_age_when_sold'], remodeled_same_as_built['SalePrice'])
ax[0].set_title('remodeling year same as built')
ax[0].set_xlabel('remodeling age when sold')
ax[0].set_ylabel('price')

ax[1].scatter(remodeled_not_same_as_built['remodeling_age_when_sold'], remodeled_not_same_as_built['SalePrice'])
ax[1].set_title('remodeling year NOT same as built')
ax[1].set_xlabel('remodeling age when sold')
ax[1].set_ylabel('price')
# -

train_origin_df['SalePrice'].mean()

fig, ax = plt.subplots(1, 1)
ax.scatter(train_for_play['OverallQual'], train_for_play['OverallCond'])


