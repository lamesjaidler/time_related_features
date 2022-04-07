import numpy as np
import pandas as pd
from time_related_features.rolling_window.std_dev import StdDevValue
from time_related_features.rolling_window.sum import SumValue, SumValueChargebacks
from ._shared import _create_data, _assert_transforms


def test_sum_value_chargebacks():
    expected_result = pd.DataFrame(
        np.array([
            [2.00000000e+00, 8.70012148e+02, 8.70012148e+02, 8.70012148e+02],
            [4.00000000e+00, 7.99158564e+02, 1.66917071e+03, 1.66917071e+03],
            [5.00000000e+00, 4.61479362e+02, 2.13065007e+03, 2.13065007e+03],
            [6.00000000e+00, 7.80529176e+02, 2.91117925e+03, 2.91117925e+03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [8.00000000e+00, 6.39921021e+02, 6.39921021e+02, 6.39921021e+02],
            [1.00000000e+00, 7.78156751e+02, 7.78156751e+02, 7.78156751e+02],
            [3.00000000e+00, 0.00000000e+00, 7.78156751e+02, 7.78156751e+02],
            [7.00000000e+00, 1.18274426e+02, 8.96431177e+02, 8.96431177e+02],
            [9.00000000e+00, 1.43353287e+02, 2.61627713e+02, 1.03978446e+03]
        ]),
        columns=[
            'eid',
            'email.sum_chargeback_txn_amt_per_email_1day',
            'email.sum_chargeback_txn_amt_per_email_7day',
            'email.sum_chargeback_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = SumValueChargebacks(
        value_col='txn_amt',
        cb_col='sim_is_fraud'
    )
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'sum_chargeback_txn_amt_per_email_'
    )


def test_sum_value():
    expected_result = pd.DataFrame(
        np.array([
            [2.00000000e+00, 8.70012148e+02, 8.70012148e+02, 8.70012148e+02],
            [4.00000000e+00, 7.99158564e+02, 1.66917071e+03, 1.66917071e+03],
            [5.00000000e+00, 4.61479362e+02, 2.13065007e+03, 2.13065007e+03],
            [6.00000000e+00, 7.80529176e+02, 2.91117925e+03, 2.91117925e+03],
            [0.00000000e+00, 8.32619846e+02, 8.32619846e+02, 8.32619846e+02],
            [8.00000000e+00, 6.39921021e+02, 6.39921021e+02, 1.47254087e+03],
            [1.00000000e+00, 7.78156751e+02, 7.78156751e+02, 7.78156751e+02],
            [3.00000000e+00, 9.78618342e+02, 1.75677509e+03, 1.75677509e+03],
            [7.00000000e+00, 1.18274426e+02, 1.87504952e+03, 1.87504952e+03],
            [9.00000000e+00, 1.43353287e+02, 1.24024606e+03, 2.01840281e+03]
        ]),
        columns=[
            'eid',
            'email.sum_txn_amt_per_email_1day',
            'email.sum_txn_amt_per_email_7day',
            'email.sum_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = SumValue(value_col='txn_amt')
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'sum_txn_amt_per_email_'
    )
