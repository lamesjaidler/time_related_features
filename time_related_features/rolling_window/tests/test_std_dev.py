import numpy as np
import pandas as pd
from time_related_features.rolling_window.std_dev import StdDevValue
from ._shared import _create_data, _assert_transforms


def test_StdDevValue():
    expected_result = pd.DataFrame(
        np.array([
            [2.,   0.,   0.,   0.],
            [4.,   0.,  35.42679202,  35.42679202],
            [5.,   0., 178.24655553, 178.24655553],
            [6.,   0., 157.33990896, 157.33990896],
            [0.,   0.,   0.,   0.],
            [8.,   0.,   0.,  96.34941211],
            [1.,   0.,   0.,   0.],
            [3.,   0., 100.23079564, 100.23079564],
            [7.,   0., 367.54760643, 367.54760643],
            [9.,   0., 399.78998869, 380.55003657]
        ]),
        columns=[
            'eid',
            'email.std_dev_txn_amt_per_email_1day',
            'email.std_dev_txn_amt_per_email_7day',
            'email.std_dev_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = StdDevValue(value_col='txn_amt')
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'std_dev_txn_amt_per_email_'
    )
