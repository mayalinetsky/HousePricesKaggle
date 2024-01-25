"""
Utilities module containing helper functions for the project
"""

from typing import Literal, Union
import numpy as np
import pandas as pd


def load_house_prices_data(source: Union[Literal['train'], Literal['test'], Literal['all']]):
    """
    Load train and\or test data.
    For information on columns, refer to data_description.txt
    """
    pass