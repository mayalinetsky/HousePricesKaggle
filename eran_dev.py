import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer

housing_train = pd.read_csv('train.csv')
housing_test = pd.read_csv('test.csv')

display(housing_train.head(2))
display(housing_test.head(2))

# +
simple_linear_model = LinearRegression()
train, validation = train_test_split(housing_train, test_size=0.2, random_state=0)
train = train.select_dtypes(include='number')
validation = validation.select_dtypes(include='number')

simple_imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')
simple_imputer.set_output(transform='pandas')
train = simple_imputer.fit_transform(train)
validation = simple_imputer.fit_transform(validation)

simple_linear_model.fit(train.drop('SalePrice', axis = 1), train['SalePrice'])
y_pred = simple_linear_model.predict(validation.drop('SalePrice', axis = 1))
y_true = np.array(validation['SalePrice'])
# print(y_pred < 0)
# print(y_true < 0)

rmse = mean_squared_log_error(y_true, y_pred)
print(rmse)
# -


