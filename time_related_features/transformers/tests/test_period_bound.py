import pytest
from time_related_features.transformers.period_bound import PeriodBoundTransformer


def test_errors_period_type():
    with pytest.raises(
            ValueError, match='`period_type` must be one of the following: second, minute, hour, day'):
        PeriodBoundTransformer(
            agg_class=None, period_type='error', periods=None,
            container_key=None, timestamp_col='timestamp'
        )
