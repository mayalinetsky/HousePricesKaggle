"""
Functions that plot graphs for different occasions.
Split from notebook for readability.
"""
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def plot_price_dist_per_year(df: pd.DataFrame):
    """
    Plot a line chart of the mean price (y-axis) per year (x-axis), with a colored area plot indicating the variance
     of prices in the year.
    """
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
