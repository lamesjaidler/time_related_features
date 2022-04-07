import pandas as pd


class UnboundTransformer:
    """
    Calculates a window aggregation across the entire dataset.

    Parameters
    ----------
    agg_class : object 
        The unbound aggregate class (from either the `exponential_window` 
        or `time_deltas` modules) to calculate.        
    container_key : str
        The field to calculate the unbound aggregate by.
    timestamp_col : str
        The timestamp columns in the dataset.
    output_names : List[str], optional
        The names of the aggregate features created. If None, a default set
        of names are generated. Defaults to None.        
    """

    def __init__(self,
                 agg_class: object,
                 container_key: str,
                 timestamp_col: str,
                 output_name=None):

        self.agg_class = agg_class
        self.container_key = container_key
        self.output_name = output_name
        # Assign params to `agg_class`
        agg_class.container_key = container_key
        agg_class.timestamp_col = timestamp_col

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
            X_ = X[X[self.container_key] == container_value]
            agg = self.agg_class.transform(
                X=X_,
                output_name=self.output_name
            )
            all_aggs.append(agg)
        X = pd.concat([X, pd.concat(all_aggs, axis=0)], axis=1)
        return X
