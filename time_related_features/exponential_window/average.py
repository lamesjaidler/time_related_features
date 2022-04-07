from typing import Union
from datetime import timedelta
import pandas as pd
from time_related_features.exponential_window._base import _Base
from time_related_features.utils.utils import _update_agg_name


class ExpWeightedAverage(_Base):
    """
    Calculates the historic exponentially weighted moving average of 
    `value_col` for a given container key.

    Parameters
    ----------
    halflife : Union[str, timedelta]
        The time unit (str or timedelta) over which an observation decays
        to half its value. 
    value_col : str
        The value column in the dataset.
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .ewm() method. Note
        that the `halflife` and `times` parameters are already set 
        internally by the class.
    """

    def __init__(self,
                 halflife: Union[str, timedelta],
                 value_col: str,
                 **kwargs):
        super().__init__(halflife=halflife, kwargs=kwargs)
        self.value_col = value_col

    def transform(self,
                  X: pd.DataFrame,
                  output_name: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Calculates the exponentially weighted moving window aggregation from 
        the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.        
        output_name : str
            The name of the output column.

        Returns
        -------
        List[Union[pd.DataFrame, pd.Series]]
            The exponentially weighted window aggregation(s).
        """

        ewm = self._return_ewm(X=X)
        agg = ewm[self.value_col].mean()
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.ew_avg_{self.value_col}_per_{self.container_key}'
        )
        return agg
