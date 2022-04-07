from time_related_features.exponential_window.average import ExpWeightedAverage
from time_related_features.transformers.unbound import UnboundTransformer
from ._shared import _create_data
import pandas as pd


def test_ExpWeightedAverage():
    expected_result = pd.Series(
        data={
            0.0: 832.619845547938,
            1.0: 778.1567509498504,
            2.0: 870.0121482468192,
            3.0: 913.0781283758596,
            4.0: 822.3238948924978,
            5.0: 645.0511464932641,
            6.0: 701.063807691501,
            7.0: 324.5936146532436,
            8.0: 650.0847869358478,
            9.0: 230.58085164053134
        },
        name='email.ew_avg_txn_amt_per_email'
    )
    expected_result.index.name = 'eid'
    X = _create_data()
    ewa = ExpWeightedAverage(
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
