from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Optional

from sklearn.base import BaseEstimator

from mapie._typing import ArrayLike, NDArray


class BaseCalibrator(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for the calibrators used in ``SplitCPRegressor``
    or ``SplitCPClassifier`` to estimate the conformity scores.

    The ``BaseCalibrator`` subclasses should have at least two methods:

    - ``fit`` : Fit the calibrator to estimate the conformity scores
      quantiles.

    -  ``predict`` : Predict the conformity score quantiles.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``predict``.
    """

    fit_attributes: List[str]
    sym: bool
    alpha: Optional[float]
    random_state: Optional[int]

    @abstractmethod
    def fit(
        self,
        X_calib: ArrayLike,
        conformity_scores_calib: NDArray,
        **kwargs,
    ) -> BaseCalibrator:
        """
        Fit the calibrator to estimate the conformity scores
        quantiles. The method can take as arguments any of :
        ``X, y, sample_weight, groups, y_pred_calib, conformity_scores_calib,
        X_train, y_train, z_train, sample_weight_train, train_index,
        X_calib, y_calib, z_calib, sample_weight_calib, calib_index``,
        any attributes of the ``SplitCP`` instance,
        or any other argument, which the user will have to pass as
        ``**calib_kwargs``.

        Parameters
        ----------
        X_calib: ArrayLike of shape (n_samples, n_features)
            Calibration data.

        conformity_scores_calib: ArrayLike of shape (n_samples,)
            Calibration conformity scores

        Returns
        -------
        BaseCalibrator
            Fitted self
        """

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        **kwargs,
    ) -> NDArray:
        """
        Predict the conformity score quantiles.
        The method can take as arguments any of : ``X, y_pred``,
        any attributes of the ``SplitCP`` instance,
        or any other argument, which the user will have to pass as
        ``**kwargs``.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Observed samples

        Returns
        -------
        NDArray of shape (n_samples,)
            Prediction
        """
