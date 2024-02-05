import pandas as pd

housing_train = pd.read_csv('train.csv')

housing_train.shape

housing_train.columns

housing_train.head()

housing_train.describe()


def count_nans(column):
    return column.isna().sum()


num_of_nans = housing_train.apply(count_nans)
num_of_nans[num_of_nans != 0]

max_nans = num_of_nans.nlargest(3).index
max_nans

max_nans = max(num_of_nans)
max_nans

# +
import matplotlib.pyplot as plt

price_change_over_years = housing_train[['YrSold', 'SalePrice']].copy().sort_values('YrSold')
price_change_over_years

# +
import matplotlib.pyplot as plt

price_sorted_by_year = housing_train[['YrSold', 'SalePrice']].copy().sort_values('YrSold')
price_sorted_by_year
# -

mean_price_per_year = price_sorted_by_year.groupby('YrSold').mean()
mean_price_per_year

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(mean_price_per_year.index, mean_price_per_year['SalePrice'])

plt.hist(housing_train['LotArea'], range=(0, 50000), bins = 20)
plt.show()

housing_train['MSSubClass'].value_counts().values

MSSubClass_counts = housing_train['MSSubClass'].value_counts().values
plt.bar(housing_train['MSSubClass'].unique(), height = MSSubClass_counts)
plt.show()

plt.hist(housing_train['OverallQual'])
plt.show()

corr_vector = housing_train.select_dtypes(include='number').corr()['SalePrice'].sort_values()
corr_vector

housing_train.loc[housing_train['YearRemodAdd'].dropna() == housing_train['YearBuilt'].dropna()]['SalePrice'].mean()

housing_train['SalePrice'].mean()

# #### Average age of never remodded houses:

2024 - housing_train.loc[housing_train['YearRemodAdd'].dropna() == housing_train['YearBuilt'].dropna()]['YearBuilt'].mean()

# #### Average age of all houses:

2024 - housing_train['YearBuilt'].mean()

# #### Average age of houses that were remodded at some point:

2024 - housing_train.loc[housing_train['YearRemodAdd'].dropna() != housing_train['YearBuilt'].dropna()]['YearBuilt'].mean()

df = housing_train.loc[housing_train['Utilities'] == 'NoSeWa']

pd.set_option('display.max_columns', None)

df

features_to_remove = ['GarageYrBlt', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GarageCars', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                           'LandSlope', 'Condition2', 'RoofMatl', 'BsmtExposure', 
                           'BsmtFinType1', 'BsmtFinType2', 'Electrical', 
                           'Functional', 'Fence', 'MiscFeature']

housing_train_filtered = housing_train.drop(features_to_remove, axis = 1)

housing_train_filtered.shape

housing_train_filtered['LowQualFinSF'].value_counts()

housing_train_filtered['ScreenPorch'].value_counts()

from sklearn.decomposition import PCA

house_train_pca = PCA()
house_train_pca.fit(housing_train)
