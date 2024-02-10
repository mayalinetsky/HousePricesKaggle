"""
Module executing the project flow.
See the model_flow.png for a diagram.
"""

import time
import numpy as np
import pandas as pd


def prepare_submission_csv(ids: np.ndarray, sale_price_prediction: np.ndarray):
    int_ids = ids.astype(int)
    submission_df = pd.DataFrame(data={"Id": int_ids, "SalePrice": sale_price_prediction})

    datetime_str = time.strftime('%Y-%m-%d--%H-%M-%S')
    submission_df.to_csv(f"predictions_{datetime_str}.csv",
                         index=False)



