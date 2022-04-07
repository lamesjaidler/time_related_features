import numpy as np
import pandas as pd
from time_related_features.transformers.period_bound import PeriodBoundTransformer
from time_related_features.rolling_window.average import AverageChargebacks, AverageTransactions, AverageValue
from ._shared import _create_data, _assert_transforms


def test_AverageChargebacks():
    expected_result = pd.DataFrame(
        np.array([
            [2., 1., 0.14285714, 0.03333333],
            [4., 1., 0.28571429, 0.06666667],
            [5., 1., 0.42857143, 0.1],
            [6., 1., 0.57142857, 0.13333333],
            [0., 0., 0., 0.],
            [8., 1., 0.14285714, 0.03333333],
            [1., 1., 0.14285714, 0.03333333],
            [3., 0., 0.14285714, 0.03333333],
            [7., 1., 0.28571429, 0.06666667],
            [9., 1., 0.28571429, 0.1]
        ]),
        columns=[
            'eid',
            'email.avg_chargeback_txn_per_email_1day',
            'email.avg_chargeback_txn_per_email_7day',
            'email.avg_chargeback_txn_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = AverageChargebacks(
        cb_col='sim_is_fraud',
    )
    X = _create_data()
    _assert_avg_event_transforms(
        agg_class, expected_result, X, 'avg_chargeback_txn_per_email_'
    )


def test_AverageTransactions():
    expected_result = pd.DataFrame(
        np.array([
            [2., 1., 0.14285714, 0.03333333],
            [4., 1., 0.28571429, 0.06666667],
            [5., 1., 0.42857143, 0.1],
            [6., 1., 0.57142857, 0.13333333],
            [0., 1., 0.14285714, 0.03333333],
            [8., 1., 0.14285714, 0.06666667],
            [1., 1., 0.14285714, 0.03333333],
            [3., 1., 0.28571429, 0.06666667],
            [7., 1., 0.42857143, 0.1],
            [9., 1., 0.42857143, 0.13333333]
        ]),
        columns=[
            'eid',
            'email.avg_num_txn_per_email_1day',
            'email.avg_num_txn_per_email_7day',
            'email.avg_num_txn_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = AverageTransactions()
    X = _create_data()
    _assert_avg_event_transforms(
        agg_class, expected_result, X, 'avg_num_txn_per_email_'
    )


def test_AverageValue():
    expected_result = pd.DataFrame(
        np.array([
            [2., 870.01214825, 870.01214825, 870.01214825],
            [4., 799.15856422, 834.58535623, 834.58535623],
            [5., 461.47936225, 710.21669157, 710.21669157],
            [6., 780.52917629, 727.79481275, 727.79481275],
            [0., 832.61984555, 832.61984555, 832.61984555],
            [8., 639.92102133, 639.92102133, 736.27043344],
            [1., 778.15675095, 778.15675095, 778.15675095],
            [3., 978.61834223, 878.38754659, 878.38754659],
            [7., 118.27442587, 625.01650635, 625.01650635],
            [9., 143.35328741, 413.41535184, 504.60070162]
        ]),
        columns=[
            'eid',
            'email.avg_txn_amt_per_email_1day',
            'email.avg_txn_amt_per_email_7day',
            'email.avg_txn_amt_per_email_30day'
        ]
    )
    expected_result.set_index('eid', inplace=True)
    agg_class = AverageValue(value_col='txn_amt')
    X = _create_data()
    _assert_transforms(
        agg_class, expected_result, X, 'avg_txn_amt_per_email_'
    )


def _assert_avg_event_transforms(agg_class, expected_result, X, col_prefix):
    # Check day aggregation
    pbt = PeriodBoundTransformer(
        agg_class=agg_class,
        periods=[1, 7, 30],
        period_type='Day',
        container_key='email',
        timestamp_col='timestamp'
    )
    X = pbt.transform(X=X)
    X.sort_values(['email', 'timestamp'], ascending=[True, True], inplace=True)
    X_ = X.iloc[:, -3:]
    pd.testing.assert_frame_equal(X_, expected_result)
    # Check hour aggregation
    pbt = PeriodBoundTransformer(
        agg_class=agg_class,
        periods=[24, 168, 720],
        period_type='Hour',
        container_key='email',
        timestamp_col='timestamp'
    )
    X = pbt.transform(X=X)
    X.sort_values(['email', 'timestamp'], ascending=[True, True], inplace=True)
    X_ = X.iloc[:, -3:] * 24
    expected_result.columns = [
        f'email.{col_prefix}24hour',
        f'email.{col_prefix}168hour',
        f'email.{col_prefix}720hour'
    ]
    pd.testing.assert_frame_equal(X_, expected_result)
    # Check minute aggregation
    pbt = PeriodBoundTransformer(
        agg_class=agg_class,
        periods=[1440, 10080, 43200],
        period_type='Minute',
        container_key='email',
        timestamp_col='timestamp'
    )
    X = pbt.transform(X=X)
    X.sort_values(['email', 'timestamp'], ascending=[True, True], inplace=True)
    X_ = X.iloc[:, -3:] * 1440
    expected_result.columns = [
        f'email.{col_prefix}1440minute',
        f'email.{col_prefix}10080minute',
        f'email.{col_prefix}43200minute'
    ]
    pd.testing.assert_frame_equal(X_, expected_result)
    # Check second aggregation
    pbt = PeriodBoundTransformer(
        agg_class=agg_class,
        periods=[86400, 604800, 2592000],
        period_type='Second',
        container_key='email',
        timestamp_col='timestamp'
    )
    X = pbt.transform(X=X)
    X.sort_values(['email', 'timestamp'], ascending=[True, True], inplace=True)
    X_ = X.iloc[:, -3:] * 86400
    expected_result.columns = [
        f'email.{col_prefix}86400second',
        f'email.{col_prefix}604800second',
        f'email.{col_prefix}2592000second'
    ]
    pd.testing.assert_frame_equal(X_, expected_result)
