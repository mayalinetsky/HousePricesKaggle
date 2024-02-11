from typing import Tuple
from dataclasses import dataclass
import pandas as pd


X_y = Tuple[pd.DataFrame, pd.Series]


@dataclass
class RawFold:
    train_raw_data: pd.DataFrame
    val_raw_data: pd.DataFrame
    test_raw_data: pd.DataFrame


@dataclass
class ProcessedFold:
    train_X_y: X_y = None
    val_X_y: X_y = None
    test_X_y: X_y = None
