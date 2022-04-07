import numpy as np
import pandas as pd
from time_related_features.transformers.period_bound import PeriodBoundTransformer


def _create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        'eid': np.linspace(0, 9, 10),
        'timestamp': [
            '01-01-2021 15:00:00', '02-01-2021 16:00:00', '03-01-2021 17:00:00', '04-01-2021 18:00:00',
            '05-01-2021 19:00:00', '06-01-2021 20:00:00', '07-01-2021 21:00:00', '08-01-2021 22:00:00',
            '09-01-2021 23:00:00', '10-01-2021 00:00:00'
        ],
        'sim_is_fraud': np.random.randint(0, 2, 10),
        'email': [f'{i}@email.com' for i in np.random.randint(0, 3, 10)],
        'ip': [f'192.168.0.{i}' for i in np.random.randint(0, 3, 10)],
        'txn_amt': np.random.uniform(0, 1000, 10),
        'name': np.random.choice(['James', 'Bill', 'Harry', 'George', 'Fred', 'Keith'], size=10)
    })
    X.set_index('eid', inplace=True)
    X['timestamp'] = pd.to_datetime(
        X['timestamp'], infer_datetime_format=True, dayfirst=True
    )
    return X


def _assert_transforms(agg_class, expected_result, X, col_prefix):
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
        X.sort_values(
            ['email', 'timestamp'], ascending=[True, True], inplace=True
        )
        X_ = X.iloc[:, -3:]
        expected_result.columns = [
            f'email.{col_prefix}{period}{period_type.lower()}' for period in periods
        ]
        pd.testing.assert_frame_equal(X_, expected_result)
