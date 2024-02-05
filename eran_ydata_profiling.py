import pandas as pd

from ydata_profiling import ProfileReport

housing_train = pd.read_csv('train.csv')

housing_train.shape

housing_train.head()

housing_train.describe()

profile = ProfileReport(housing_train, title = 'Housing report after pillow upgrade')

profile.to_file('housing_profile_report_after_pillow_upgrade_test_leehee.html')

features_to_remove = ['GarageYrBlt', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GarageCars', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                           'LandSlope', 'Condition2', 'RoofMatl', 'BsmtExposure', 
                           'BsmtFinType1', 'BsmtFinType2', 'Electrical', 
                           'Functional', 'Fence', 'MiscFeature']

housing_train_filtered = housing_train.drop(features_to_remove, axis = 1)

housing_train_filtered.shape

profile = ProfileReport(housing_train_filtered, title = 'Housing report after Yair features filter')

profile.to_file('housing_profile_report_after_Yair_features_filter.html')

features_to_remove = ['GarageYrBlt', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GarageCars', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                           'LandSlope', 'Condition2', 'RoofMatl', 'BsmtExposure', 
                           'BsmtFinType1', 'BsmtFinType2', 'Electrical', 
                           'Functional', 'Fence', 'MiscFeature']
