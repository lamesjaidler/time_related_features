from time_related_features.rolling_window._base import _Base
from time_related_features.utils.utils import _update_agg_name
from typing import List, Union
import pandas as pd


class StdDevValue(_Base):
    """
    Calculates the standard deviation of the `value_col` for the period given, 
    for a given container key.

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
        agg = rw[self.value_col].std(ddof=0)
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.std_dev_{self.value_col}_per_{self.container_key}_{period}{self.period_type}'
        )
        return [agg]
