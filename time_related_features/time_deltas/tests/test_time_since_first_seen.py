from time_related_features.time_deltas.time_since_first_seen import TimeSinceFirstSeen
import pandas as pd
import numpy as np
from ._shared import _create_data, _assert_transform


def test_time_since_first_seen():
    expected_result = pd.DataFrame(
        np.array([
            [2.0, pd.Timedelta('0 days 00:00:00')],
            [4.0, pd.Timedelta('2 days 02:00:00')],
            [5.0, pd.Timedelta('3 days 03:00:00')],
            [6.0, pd.Timedelta('4 days 04:00:00')],
            [0.0, pd.Timedelta('0 days 00:00:00')],
            [8.0, pd.Timedelta('8 days 08:00:00')],
            [1.0, pd.Timedelta('0 days 00:00:00')],
            [3.0, pd.Timedelta('2 days 02:00:00')],
            [7.0, pd.Timedelta('6 days 06:00:00')],
            [9.0, pd.Timedelta('7 days 08:00:00')]],
            dtype=object),
        columns=['eid', 'email.time_since_txn_first_seen_by_email']
    )
    expected_result.set_index('eid', inplace=True)
    expected_result = expected_result.squeeze()
    X = _create_data()
    agg_class = TimeSinceFirstSeen()
    _assert_transform(agg_class, X, expected_result)
