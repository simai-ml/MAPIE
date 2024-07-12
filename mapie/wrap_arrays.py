from typing import Union
import numpy as np
import pandas as pd
from mapie._typing import NDArray


class wrap_ndarray_and_dataframe:

    def __init__(self, X_array: Union[NDArray | pd.DataFrame]):
        """
            This class is a wrapper for numpy arrays and pandas DataFrames.
            It is used to handle the indexing access to the data
            in a consistent way.

        Parameters
        ----------
        X_array: Union[NDArray | pd.DataFrame]
            The data to wrap, either a numpy array or a pandas DataFrame.
        """
        self.X_array = X_array
        if isinstance(X_array, pd.DataFrame):
            self.X_array = pd.DataFrame(X_array, columns=X_array.columns)
            self.X_array = self.X_array.astype(self.X_array.dtypes.to_dict())

    def __getitem__(self, i: int):
        """
            This method is used to handle the indexing access to X_array.

        Parameters
        ----------
        i: int
            Index to access.

        Returns
        -------
        NDArray
            The data at index i.
        """
        if isinstance(self.X_array, pd.DataFrame):
            return self.X_array.iloc[i].values
        elif isinstance(self.X_array, np.ndarray):
            return self.X_array[i]
        else:
            raise ValueError("Input must be a numpy array or pandas DataFrame")
