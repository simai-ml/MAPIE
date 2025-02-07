from __future__ import annotations

from typing import List, cast

import numpy as np
from sklearn.utils.validation import _num_samples

from mapie._typing import ArrayLike, NDArray
from mapie.future.calibrators.base import BaseCalibrator
from .ccp.utils import check_required_arguments
from mapie.conformity_scores.interface import BaseConformityScore


class StandardCalibrator(BaseCalibrator):
    """
    Calibrator used to get the standard conformal prediciton. It is strictly
    equivalent to ``MapieRegressor`` with ``method='base'``.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``predict``.

    q_up_: float
        Calibration fitting results, used to build the upper bound of the
        prediction intervals. It correspond to the quantile of the calibration
        conformity scores.

    q_low_: float
        Same as q_up_, but for the lower bound
    """
    fit_attributes: List[str] = ["q_up_", "q_low_"]

    def __init__(self) -> None:
        return

    def fit(
        self,
        X_calib: ArrayLike,
        conformity_scores_calib: NDArray,
        allow_infinite_bounds: bool = False,
        **kwargs,
    ) -> BaseCalibrator:
        """
        Fit the calibrator instance

        Parameters
        ----------
        X_calib: ArrayLike of shape (n_samples, n_features)
            Calibration data.

        conformity_scores_calib: ArrayLike of shape (n_samples,)
            Calibration conformity scores

        allow_infinite_bounds: bool
            Allow infinite prediction intervals to be produced.
        """
        check_required_arguments(self.alpha)
        self.alpha = cast(float, self.alpha)

        # TODO: Partial copy paste of the BaseConformityScore.get_bounds method
        if self.sym:
            alpha_ref = 1-self.alpha
            quantile_ref = BaseConformityScore.get_quantile(
                conformity_scores_calib[..., np.newaxis],
                np.array([alpha_ref]), axis=0
            )[0, 0]
            self.q_low_, self.q_up_ = -quantile_ref, quantile_ref

        else:
            alpha_low, alpha_up = self.alpha/2, 1 - self.alpha/2

            self.q_low_ = BaseConformityScore.get_quantile(
                conformity_scores_calib[..., np.newaxis],
                np.array([alpha_low]), axis=0, reversed=True,
                unbounded=allow_infinite_bounds
            )[0, 0]
            self.q_up_ = BaseConformityScore.get_quantile(
                conformity_scores_calib[..., np.newaxis],
                np.array([alpha_up]), axis=0,
                unbounded=allow_infinite_bounds
            )[0, 0]

        return self

    def predict(
        self,
        X: ArrayLike,
        **kwargs,
    ) -> NDArray:
        """
        Predict the conformity scores estimation

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        Returns
        -------
        NDArray
            prediction
        """
        return np.ones((_num_samples(X), 2)) * np.array([
            self.q_low_, self.q_up_
        ])
