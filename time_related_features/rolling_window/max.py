from time_related_features.rolling_window._base import _Base
from time_related_features.utils.utils import _update_agg_name
import pandas as pd
from typing import List, Union


class MaxValueChargebacks(_Base):
    """
    Calculates the maximum of the `value_col` for chargeback transactions, 
    for the period given, for a given container key.

    Parameters
    ----------
    cb_col : str
        The chargeback column in the dataset.
    value_col : str
        The value column in the dataset.
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .rolling() method. Note
        that the `window` and `on` parameters are already set internally by the
        class.
    """

    def __init__(self,
                 cb_col: str,
                 value_col: str,
                 **kwargs):

        super().__init__(kwargs=kwargs)
        self.cb_col = cb_col
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
        X = X.assign(value_cb_only=X[self.cb_col] * X[self.value_col])
        rw = self._return_rw(X=X, period=period)
        agg = rw['value_cb_only'].max()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.max_chargeback_{self.value_col}_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]


class MaxValue(_Base):
    """
    Calculates the maximum of the `value_col` for the period given, for a given
    container key.

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
        agg = rw[self.value_col].max()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.max_{self.value_col}_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]
