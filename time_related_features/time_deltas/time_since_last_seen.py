"""
Calculates the time since an event was last seen for a given container key.
"""
from time_related_features.time_deltas._base import _Base
from time_related_features.utils.utils import _update_agg_name
import pandas as pd
from typing import Union


class TimeSinceLastSeen(_Base):
    """
    Calculates the time since an event was last seen for a given container key.
    """

    def __init__(self):
        pass

    def transform(self,
                  X: pd.DataFrame,
                  output_name: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method for calculating the time delta feature from the
        dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.        
        output_name : str
            The name of the output column.

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
            The time delta feature(s).
        """
        X = X.sort_values(self.timestamp_col, ascending=True)
        agg = X[self.timestamp_col].diff(1)
        agg.fillna(pd.Timedelta(0), inplace=True)
        agg = _update_agg_name(
            agg=agg, output_name=output_name,
            default_name=f'{self.container_key}.time_since_txn_last_seen_by_{self.container_key}'
        )
        return agg
