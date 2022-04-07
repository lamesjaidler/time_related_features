from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union

PERIOD_TYPE_MAPPING = {
    'second': 's',
    'minute': 'min',
    'hour': 'H',
    'day': 'D',
}


class _Base(ABC):
    """
    Base class for rolling window aggregations.

    Parameters
    ----------    
    **kwargs: dict
        Any keyword arguments to pass to the Pandas .rolling() method. Note
        that the `window` and `on` parameters are already set internally by the
        class.
    """

    def __init__(self, kwargs):
        self.kwargs = kwargs
        # Attributes updated when `PeriodBound.fit()` method runs:
        self.period_type = None
        self.container_key = None
        self.timestamp_col = None

    @abstractmethod
    def transform(X: pd.DataFrame,
                  period: float,
                  output_name: str) -> List[Union[pd.DataFrame, pd.Series]]:
        """
        Abstract method for calculating the rolling window aggregation from the
        dataset.

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
        pass

    def _return_rw(self,
                   X: pd.DataFrame,
                   period: float) -> pd.core.window.rolling.Rolling:
        """
        Returns the Pandas rolling window object.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.
        period : float
            The rolling window period.

        Returns
        -------
        pd.core.window.rolling.Rolling
            The Pandas rolling window object.
        """

        return X.rolling(
            window=f'{period}{PERIOD_TYPE_MAPPING[self.period_type]}',
            on=self.timestamp_col,
            **self.kwargs
        )
