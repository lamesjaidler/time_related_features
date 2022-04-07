from time_related_features.exponential_window.std_dev import ExpWeightedStdDev
from time_related_features.transformers.unbound import UnboundTransformer
from ._shared import _create_data
import pandas as pd
import numpy as np


def test_ExpWeightedStdDev():
    expected_result = pd.Series(
        data={
            0.0: np.nan,
            1.0: np.nan,
            2.0: np.nan,
            3.0: 141.74775056359428,
            4.0: 50.10104973905146,
            5.0: 238.28663390807557,
            6.0: 187.40395870903666,
            7.0: 525.8897788636137,
            8.0: 136.25864533292938,
            9.0: 397.7642421926121
        },
        name='email.ew_std_dev_txn_amt_per_email'
    )
    expected_result.index.name = 'eid'
    X = _create_data()
    ewa = ExpWeightedStdDev(
        halflife='2d',
        value_col='txn_amt'
    )
    ut = UnboundTransformer(
        agg_class=ewa,
        container_key='email',
        timestamp_col='timestamp'
    )
    X = ut.transform(X)
    X = X.iloc[:, -1]
    pd.testing.assert_series_equal(X, expected_result)
