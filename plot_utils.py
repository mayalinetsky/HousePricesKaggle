"""
Functions that plot graphs for different occasions.
Split from notebook for readability.
"""
import warnings

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from preprocessing import _convert_nan_to_str


def plot_price_dist_per_year(df: pd.DataFrame):
    """
    Plot a line chart of the mean price (y-axis) per year (x-axis), with a colored area plot indicating the variance
     of prices in the year.
    """
    warnings.filterwarnings("ignore", "use_inf_as_na")
    price_sorted_by_year = df[['YrSold', 'SalePrice']].copy().sort_values('YrSold')
    prices_per_year = price_sorted_by_year.groupby('YrSold').apply(list)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(mean_price_per_year.index, mean_price_per_year['SalePrice'])

    sns.lineplot(data=price_sorted_by_year, x="YrSold", y="SalePrice", errorbar=('ci', 75))
    sns.despine()

    fig.suptitle("Distribution of House Prices Over the Years")
    ax.set_title("Mean Price and 75% Confidence Interval")
    plt.show()


def plot_num_missing_values(col_to_nans: pd.Series):
    """
    Plot a horizontal bar chart of the number of missing values per column,
    sorted from lowest to highest.

    :param col_to_nans: series with index of columns names with values equal
    to the number of missing values in the column
    """
    fig, ax = plt.subplots()
    y_pos = np.arange(len(col_to_nans))
    ax.barh(y_pos, col_to_nans.values, align='center')
    ax.set_yticks(y_pos, labels=col_to_nans.index)
    ax.set_title('Number of Missing Values Per Column')
    plt.show()


def plot_numeric_features_correlation_to_target(corr_vector: pd.Series):
    plt.figure(figsize=(10, 8))
    corr_vector.plot(kind='barh')

    plt.title('Correlation to Sale Price of Numerical Features');
    plt.xlabel("Correlation to Sale Price");
    plt.show()


def plot_column_histograms(df: pd.DataFrame):
    """
    Plot a histogram for each column in the dataframe.
    """
    df = _convert_nan_to_str(df, inplace=False)
    fig, axes = plt.subplots(ncols=9, nrows=9, figsize=(20, 16))
    for col_index, col in enumerate(df.columns):
        ax_col_index = col_index % 9
        ax_row_index = col_index // 9
        ax_to_plot = axes[ax_row_index, ax_col_index]
        ax_to_plot.set_title(col)
        df[col].hist(ax=ax_to_plot)

    plt.suptitle("Feature Histograms")
    plt.tight_layout()
    plt.show()


def plot_head_and_tail_categorical_corr_to_target(sorted_correlation: pd.Series):
    sorted_correlation.head(10).plot(kind='barh')
    plt.title('Top Positive Correlation to Sale Price of Categorical Features (OneHotEncoding)');
    plt.xlabel("Correlation to Sale Price");
    plt.show()

    sorted_correlation.tail(10).plot(kind='barh')
    plt.title('Top Negative Correlation to Sale Price of Categorical Features (OneHotEncoding)');
    plt.xlabel("Correlation to Sale Price");
    plt.show()


def plot_mean_price_and_stddev_per_category(df: pd.DataFrame):
    fig, axes = plt.subplots(11, 4, figsize=(12, 35))
    for i, column in enumerate(df.select_dtypes(include=['object']).columns):
        target_mean = df.groupby(column)['SalePrice'].mean()

        target_std = df.groupby(column)['SalePrice'].std()

        cur_ax = axes[i // 4, i % 4]
        target_mean.plot(kind='bar', ax=cur_ax, yerr=target_std, capsize=5)
        cur_ax.set_title(column)
    plt.tight_layout()
    plt.show()


def plot_number_of_sales_and_prices_across_time(df: pd.DataFrame):
    """
    TODO split function into smaller ones
    """
    # Looking for seasonality in number of sales
    sales_grouped = df.groupby(['YrSold', 'MoSold']).size()
    sales_grouped_reset = sales_grouped.reset_index(name='Count')
    sales_grouped_reset['Year-Month'] = sales_grouped_reset['YrSold'].astype(str) + '-' + sales_grouped_reset[
        'MoSold'].astype(str)
    plt.figure(figsize=(12, 3))
    plt.plot(sales_grouped_reset['Year-Month'], sales_grouped_reset['Count'])
    plt.xticks(rotation=45)
    plt.title('Sales Seasonality')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Sales')
    plt.grid(True)

    # looking for seasonality in sale prices
    price_grouped = df.groupby(['YrSold', 'MoSold'])['SalePrice'].mean()
    price_grouped_reset = price_grouped.reset_index(name='AvgPrice')
    price_grouped_reset['Year-Month'] = price_grouped_reset['YrSold'].astype(str) + '-' + price_grouped_reset[
        'MoSold'].astype(str)
    plt.figure(figsize=(12, 3))
    plt.plot(price_grouped_reset['Year-Month'], price_grouped_reset['AvgPrice'])
    plt.xticks(rotation=45)
    plt.title('Price Seasonality')
    plt.xlabel('Year-Month')
    plt.ylabel('Average Sale Price')
    plt.xlim('2006-1', '2010-6')
    plt.ylim(150000)
    plt.grid(True)

    # Looking for correlation in both seasonality patterns
    fig, ax1 = plt.subplots(figsize=(12, 3))

    ax1.plot(sales_grouped_reset['Year-Month'], sales_grouped_reset['Count'], label='Sales Count', color='blue')
    ax1.set_xlabel('Year-Month')
    ax1.set_ylabel('Number of Sales', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(price_grouped_reset['Year-Month'], price_grouped_reset['AvgPrice'], label='Average Price', color='red')
    ax2.set_ylabel('Average Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.set_ylim(150000)
    plt.title('Sales and Average Price Seasonality')
    plt.grid(True)
    plt.show()
