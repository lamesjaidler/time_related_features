import numpy as np
import pandas as pd
from time_related_features.rolling_window.max import MaxValue, MaxValueChargebacks
from ._shared import _create_data, _assert_transforms


def test_MaxValueChargebacks():
    expected_result = pd.DataFrame(
        np.array([
            [2., 870.01214825, 870.01214825, 870.01214825],
            [4., 799.15856422, 870.01214825, 870.01214825],
            [5., 461.47936225, 870.01214825, 870.01214825],
            [6., 780.52917629, 870.01214825, 870.01214825],
            [0.,   0.,   0.,   0.],
            [8., 639.92102133, 639.92102133, 639.92102133],
            [1., 778.15675095, 778.15675095, 778.15675095],
            [3.,   0., 778.15675095, 778.15675095],
            [7., 118.27442587, 778.15675095, 778.15675095],
            [9., 143.35328741, 143.35328741, 778.15675095]
        ]),
        columns=[
            'eid',
            'email.max_chargeback_txn_amt_per_email_1day',
            'email.max_chargeback_txn_amt_per_email_7day',
            'email.max_chargeback_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = MaxValueChargebacks(
        value_col='txn_amt',
        cb_col='sim_is_fraud',
    )
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'max_chargeback_txn_amt_per_email_'
    )


def test_MaxValue():
    expected_result = pd.DataFrame(
        np.array([
            [2., 870.01214825, 870.01214825, 870.01214825],
            [4., 799.15856422, 870.01214825, 870.01214825],
            [5., 461.47936225, 870.01214825, 870.01214825],
            [6., 780.52917629, 870.01214825, 870.01214825],
            [0., 832.61984555, 832.61984555, 832.61984555],
            [8., 639.92102133, 639.92102133, 832.61984555],
            [1., 778.15675095, 778.15675095, 778.15675095],
            [3., 978.61834223, 978.61834223, 978.61834223],
            [7., 118.27442587, 978.61834223, 978.61834223],
            [9., 143.35328741, 978.61834223, 978.61834223]
        ]),
        columns=[
            'eid',
            'email.max_txn_amt_per_email_1day',
            'email.max_txn_amt_per_email_7day',
            'email.max_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = MaxValue(value_col='txn_amt')
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'max_txn_amt_per_email_'
    )
