"""

"""
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer


def rmse_log(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))


rmse_log_scorer = make_scorer(rmse_log, greater_is_better=False)
