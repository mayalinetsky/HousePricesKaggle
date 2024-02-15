"""
Split raw data into folds
"""
import pandas as pd
from constants import *
from custom_types import RawFold


class BaseTrainValTestSplitter:
    """
    Produces only 1 fold with the validation set being the train set.
    (This is used for the baseline)
    """
    def __init__(self, *args, **kwargs):
        pass

    def split(self, train_rawdata, test_rawdata) -> list[RawFold]:
        return [RawFold(train_rawdata, train_rawdata, test_rawdata)]


class AllUntilMonthSplitter(BaseTrainValTestSplitter):
    """
    For each month-year in the test set,
    the validation set will be taken from raw training samples with the same month and year,
    and the training set will be all raw training samples until that month and year.
    """
    def __init__(self):
        super().__init__()

    def split(self, train_rawdata, test_rawdata) -> list[RawFold]:
        train_rawdata_dt = self._convert_month_year_to_dt(train_rawdata)
        test_rawdata_dt = self._convert_month_year_to_dt(test_rawdata)

        test_unique_dts = test_rawdata_dt['dt'].unique().tolist()
        all_folds = []

        for month_time_frame in test_unique_dts:
            test_rawdata_in_frame = test_rawdata[test_rawdata_dt['dt'] == month_time_frame]

            val_rawdata_in_frame = train_rawdata[train_rawdata_dt['dt'] == month_time_frame]

            train_rawdata_until_frame = train_rawdata[train_rawdata_dt['dt'] < month_time_frame]

            if len(train_rawdata_until_frame) == 0:
                # this can happen for January 2006, because both the raw training set and test set start from this month
                index_cuttoff = int(0.8 * len(val_rawdata_in_frame))
                train_rawdata_until_frame = val_rawdata_in_frame.iloc[:index_cuttoff]
                val_rawdata_in_frame = val_rawdata_in_frame.iloc[index_cuttoff:]

            all_folds.append(RawFold(train_rawdata_until_frame, val_rawdata_in_frame, test_rawdata_in_frame))

        return all_folds

    @staticmethod
    def _convert_month_year_to_dt(df: pd.DataFrame):
        df = df.copy()
        df['dt'] = pd.to_datetime(df[YrSold].astype(str) + df[MoSold].astype(str), format='%Y%m')
        return df
