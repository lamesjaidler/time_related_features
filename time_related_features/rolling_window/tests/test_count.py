import numpy as np
import pandas as pd
from time_related_features.rolling_window.count import CountDistinctChargebacks, CountDistinctCrossContainerFields, CountDistinctTransactions
from time_related_features.transformers.period_bound import PeriodBoundTransformer
from ._shared import _create_data, _assert_transforms


def test_CountDistinctChargebacks():
    expected_result = pd.DataFrame(
        np.array([
            [2., 1., 1., 1.],
            [4., 1., 2., 2.],
            [5., 1., 3., 3.],
            [6., 1., 4., 4.],
            [0., 0., 0., 0.],
            [8., 1., 1., 1.],
            [1., 1., 1., 1.],
            [3., 0., 1., 1.],
            [7., 1., 2., 2.],
            [9., 1., 2., 3.]
        ]),
        columns=[
            'eid',
            'email.num_distinct_chargeback_txn_per_email_1day',
            'email.num_distinct_chargeback_txn_per_email_7day',
            'email.num_distinct_chargeback_txn_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = CountDistinctChargebacks(cb_col='sim_is_fraud')
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'num_distinct_chargeback_txn_per_email_'
    )


def test_CountDistinctTransactions():
    expected_result = pd.DataFrame(
        np.array([
            [2., 1., 1., 1.],
            [4., 1., 2., 2.],
            [5., 1., 3., 3.],
            [6., 1., 4., 4.],
            [0., 1., 1., 1.],
            [8., 1., 1., 2.],
            [1., 1., 1., 1.],
            [3., 1., 2., 2.],
            [7., 1., 3., 3.],
            [9., 1., 3., 4.]
        ]),
        columns=[
            'eid',
            'email.num_distinct_txn_per_email_1day',
            'email.num_distinct_txn_per_email_7day',
            'email.num_distinct_txn_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = CountDistinctTransactions()
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'num_distinct_txn_per_email_'
    )


def test_CountDistinctCrossContainerFields():
    expected_result = pd.DataFrame(
        np.array([
            [2., 1., 1., 1., 1., 1., 1.],
            [4., 1., 1., 1., 2., 1., 2.],
            [5., 1., 1., 1., 3., 1., 3.],
            [6., 1., 1., 2., 3., 2., 3.],
            [0., 1., 1., 1., 1., 1., 1.],
            [8., 1., 1., 1., 1., 2., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [3., 1., 1., 2., 2., 2., 2.],
            [7., 1., 1., 2., 2., 2., 2.],
            [9., 1., 1., 2., 3., 2., 3.]
        ]),
        columns=[
            'eid',
            'email.num_distinct_ip_per_email_1day',
            'email.num_distinct_name_per_email_1day',
            'email.num_distinct_ip_per_email_7day',
            'email.num_distinct_name_per_email_7day',
            'email.num_distinct_ip_per_email_30day',
            'email.num_distinct_name_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = CountDistinctCrossContainerFields(
        cross_container_fields=['ip', 'name']
    )
    X = _create_data()
    periods_list = [
        [1, 7, 30],
        [24, 168, 720],
        [1440, 10080, 43200],
        [86400, 604800, 2592000],
    ]
    period_type_list = [
        'Day',
        'Hour',
        'Minute',
        'Second'
    ]
    combs = zip(periods_list, period_type_list)
    for periods, period_type in combs:
        pbt = PeriodBoundTransformer(
            agg_class=agg_class,
            periods=periods,
            period_type=period_type,
            container_key='email',
            timestamp_col='timestamp'
        )
        X = pbt.transform(X=X)
        X.sort_values(['email', 'timestamp'], ascending=[
                      True, True], inplace=True)
        X_ = X.iloc[:, -6:]
        expected_result.columns = [
            f'{f}{period}{period_type.lower()}' for period in periods for f in [
                'email.num_distinct_ip_per_email_', 'email.num_distinct_name_per_email_']
        ]
        pd.testing.assert_frame_equal(X_, expected_result)
