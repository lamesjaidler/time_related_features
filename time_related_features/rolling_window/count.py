from time_related_features.rolling_window._base import _Base
from time_related_features.utils.utils import _update_agg_name
from sklearn import preprocessing
import numpy as np
import pandas as pd
from typing import List, Union


class CountDistinctChargebacks(_Base):
    """
    Calculates the number of distinct chargeback transactions for the 
    period given, for a given container key.

    Parameters
    ----------
    cb_col : str
        The chargeback column in the dataset.
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .rolling() method. Note
        that the `window` and `on` parameters are already set internally by the
        class.
    """

    def __init__(self,
                 cb_col: str,
                 **kwargs):
        super().__init__(kwargs=kwargs)
        self.cb_col = cb_col

    def transform(self,
                  X: pd.DataFrame,
                  period: float,
                  output_name: str) -> List[Union[pd.DataFrame, pd.Series]]:
        """
        Calculates the rolling window aggregation from the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.
        period : float
            The rolling window period.
        output_name : str
            The name of the output column.

        Returns
        -------
        List[Union[pd.DataFrame, pd.Series]]
            List containing the rolling window aggregation(s).
        """

        rw = self._return_rw(X=X, period=period)
        agg = rw[self.cb_col].sum()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.num_distinct_chargeback_txn_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]


class CountDistinctTransactions(_Base):
    """
    Calculates the number of distinct transactions for the period given, for a
    given container key.    

    Parameters
    ----------
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .rolling() method. Note
        that the `window` and `on` parameters are already set internally by the
        class.
    """

    def __init__(self,
                 **kwargs):
        super().__init__(kwargs=kwargs)

    def transform(self,
                  X: pd.DataFrame,
                  period: float,
                  output_name: str) -> List[Union[pd.DataFrame, pd.Series]]:
        """
        Calculates the rolling window aggregation from the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.
        period : float
            The rolling window period.
        output_name : str
            The name of the output column.

        Returns
        -------
        List[Union[pd.DataFrame, pd.Series]]
            List containing the rolling window aggregation(s).
        """

        X = X.assign(event_flag=1)
        rw = self._return_rw(X=X, period=period)
        agg = rw['event_flag'].count()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.num_distinct_txn_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]


class CountDistinctCrossContainerFields(_Base):
    def __init__(self,
                 cross_container_fields: List[str],
                 **kwargs):
        """
        Calculates the number of distinct values for each field in the 
        `cross_container_fields` list for the period given, for a given 
        container key.

        Parameters
        ----------
        cross_container_fields : List[str]
            The cross-container column names.
        **kwargs: dict
            Any keyword arguments to pass to the Pandas .rolling() method. Note
            that the `window` and `on` parameters are already set internally by the
            class.
        """

        super().__init__(kwargs=kwargs)
        self.cross_container_fields = cross_container_fields

    def transform(self,
                  X: pd.DataFrame,
                  period: float,
                  output_name: str) -> List[Union[pd.DataFrame, pd.Series]]:
        """
        Calculates the rolling window aggregation from the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.
        period : float
            The rolling window period.
        output_name : str
            The name of the output column.

        Returns
        -------
        List[Union[pd.DataFrame, pd.Series]]
            List containing the rolling window aggregation(s).
        """

        aggs = []
        for cross_container_field in self.cross_container_fields:
            le = preprocessing.LabelEncoder()
            X = X.assign(le_field=le.fit_transform(X[cross_container_field]))
            rw = self._return_rw(X=X, period=period)
            _agg = rw['le_field'].apply(lambda x: np.unique(x).shape[0])
            _agg = _update_agg_name(
                agg=_agg, output_name=output_name,
                default_name=f'{self.container_key}.num_distinct_{cross_container_field}_per_{self.container_key}_{period}{self.period_type}'
            )
            aggs.append(_agg)
        agg = pd.concat(aggs, axis=1)
        return [agg]
