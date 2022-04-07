from typing import Union, List
import pandas as pd
from abc import ABC, abstractmethod
from datetime import timedelta


class _Base(ABC):
    """
    Base class for exponentially weighted moving window aggregations.

    Parameters
    ----------
    halflife : Union[str, timedelta]
        The time unit (str or timedelta) over which an observation decays
        to half its value. 
    kwargs : dict
        Any keyword arguments to pass to the Pandas .ewm() method. Note
        that the `halflife` and `times` parameters are already set 
        internally by the class.
    """

    def __init__(self,
                 halflife: Union[str, timedelta],
                 kwargs: dict):
        self.halflife = halflife
        self.kwargs = kwargs
        # Attributes updated when `Unbound.transform()` method runs:
        self.timestamp_col = None
        self.container_key = None

    @abstractmethod
    def transform(X: pd.DataFrame,
                  output_name: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method for calculating the exponentially weighted moving 
        window aggregation from the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.        
        output_name : str
            The name of the output column.

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
            The exponentially weighted window aggregation(s).
        """
        pass

    def _return_ewm(self,
                    X: pd.DataFrame) -> pd.core.window.ewm.ExponentialMovingWindow:
        """
        Returns the Pandas exponentially weighted moving window object.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.        

        Returns
        -------
        pd.core.window.ewm.ExponentialMovingWindow
            The Pandas exponentially weighted moving window object.
        """

        return X.ewm(
            halflife=self.halflife,
            times=self.timestamp_col,
            **self.kwargs
        )
