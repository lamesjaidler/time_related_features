import pandas as pd
import numpy as np
from time_related_features.transformers.unbound import UnboundTransformer


def _create_data():
    X = pd.DataFrame(
        np.array([
            [2.0, '0@email.com', '2021-01-03 17:00:00'],
            [4.0, '0@email.com', '2021-01-05 19:00:00'],
            [5.0, '0@email.com', '2021-01-06 20:00:00'],
            [6.0, '0@email.com', '2021-01-07 21:00:00'],
            [0.0, '1@email.com', '2021-01-01 15:00:00'],
            [8.0, '1@email.com', '2021-01-09 23:00:00'],
            [1.0, '2@email.com', '2021-01-02 16:00:00'],
            [3.0, '2@email.com', '2021-01-04 18:00:00'],
            [7.0, '2@email.com', '2021-01-08 22:00:00'],
            [9.0, '2@email.com', '2021-01-10 00:00:00']],
            dtype=object),
        columns=['eid', 'email', 'timestamp'])
    X['timestamp'] = pd.to_datetime(
        X['timestamp'], infer_datetime_format=True,
    )
    X.set_index('eid', inplace=True)
    return X


def _assert_transform(agg_class, X, expected_result):
    td = UnboundTransformer(
        agg_class=agg_class,
        container_key='email',
        timestamp_col='timestamp'
    )
    X = td.transform(X)
    pd.testing.assert_series_equal(X.iloc[:, -1], expected_result)
