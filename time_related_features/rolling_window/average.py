from time_related_features.rolling_window._base import _Base
from time_related_features.utils.utils import _update_agg_name
import pandas as pd
from typing import List, Union


class AverageChargebacks(_Base):
    """
    Calculates the average number of chargebacks for the period and period
    type given, for a given container key (e.g. if the period_type is 'Second'
    and the period is 60, then the average number of chargebacks per second for
    the last 60 seconds is calculated).

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
        agg = rw[self.cb_col].sum() / period
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.avg_chargeback_txn_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]


class AverageTransactions(_Base):
    """
    Calculates the average number of transactions for the period and period
    type given, for a given container key (e.g. if the period_type is 'Second'
    and the period is 60, then the average number of transactions per second
    for the last 60 seconds is calculated).

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
        agg = rw['event_flag'].count() / period
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.avg_num_txn_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]


class AverageValue(_Base):
    """
    Calculates the average of the `value_col` for the period given, for a given
    container key (e.g. if the period_type is 'Second' and the period is 60,
    then the average of `value_col` in the last 60 seconds is calculated).

    Parameters
    ----------
    value_col : str
        The value column in the dataset.
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .rolling() method. Note
        that the `window` and `on` parameters are already set internally by the
        class.
    """

    def __init__(self,
                 value_col: str,
                 **kwargs):
        super().__init__(kwargs=kwargs)
        self.value_col = value_col

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
        agg = rw[self.value_col].mean()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.avg_{self.value_col}_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]
