"""
Split raw data into folds
"""
from custom_types import RawFold


class BaseTrainValTestSplitter:
    """
    Produces only 1 fold with the validation set being like the train set.
    """
    def __init__(self, *args, **kwargs):
        pass

    def split(self, train_rawdata, test_rawdata) -> list[RawFold]:
        return [RawFold(train_rawdata, train_rawdata, test_rawdata)]
