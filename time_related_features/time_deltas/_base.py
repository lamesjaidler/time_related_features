from typing import Union, List
import pandas as pd
from abc import ABC, abstractmethod


class _Base(ABC):
    def __init__(self):
        # Attributes updated when `Unbound.transform()` method runs:
        self.timestamp_col = None
        self.container_key = None

    @abstractmethod
    def transform(X: pd.DataFrame,
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
        pass
