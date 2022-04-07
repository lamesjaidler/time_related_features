import pandas as pd
from typing import List
from time_related_features.rolling_window._base import PERIOD_TYPE_MAPPING


class PeriodBoundTransformer:
    """
    Calculates a window aggregation across a given set of periods.
    """

    def __init__(self,
                 agg_class: object,
                 period_type: str,
                 periods: List[float],
                 container_key: str,
                 timestamp_col: str,
                 output_names=None):
        """        
        Parameters
        ----------
        agg_class : object 
            The rolling window class (from the `rolling_window` module) to 
            calculate.
        period_type : str
            The period type to calculate the rolling window aggregate over. Can
            be either 'second', 'minute', 'hour' or 'day'.
        periods : List[float]
            The periods to calculate the rolling window aggregate over.
        container_key : str
            The field to calculate the rolling window aggregate by.
        timestamp_col : str
            The timestamp columns in the dataset.
        output_names : List[str], optional
            The names of the aggregate features created. If None, a default set
            of names are generated. Defaults to None.

        Raises
        ------
        ValueError
            `period_type` must be either 'second', 'minute', 'hour' or 'day'.
        """

        self.agg_class = agg_class
        self.period_type = period_type.lower()
        if self.period_type not in PERIOD_TYPE_MAPPING.keys():
            raise ValueError(
                f'`period_type` must be one of the following: {", ".join(list(PERIOD_TYPE_MAPPING.keys()))}'
            )
        self.periods = periods
        self.container_key = container_key
        if output_names is None:
            self.output_names = [None] * len(periods)
        else:
            self.output_names = output_names
        self.agg_class.period_type = self.period_type
        self.agg_class.container_key = self.container_key
        self.agg_class.timestamp_col = timestamp_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the unbound aggregation.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.

        Returns
        -------
        pd.DataFrame
            Original dataset + unbound aggregate.
        """

        all_aggs = []
        container_values = X[self.container_key].unique()
        for container_value in container_values:
            aggs = []
            X_ = X[X[self.container_key] == container_value]
            for period, output_name in zip(self.periods, self.output_names):
                agg = self.agg_class.transform(
                    X=X_,
                    period=period,
                    output_name=output_name
                )
                aggs.extend(agg)
            container_value_aggs = pd.concat(aggs, axis=1)
            all_aggs.append(container_value_aggs)
        X = pd.concat([X, pd.concat(all_aggs, axis=0)], axis=1)
        return X
