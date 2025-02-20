from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import PredefinedSplit, ShuffleSplit

from mapie._typing import ArrayLike, NDArray
from mapie.future.calibrators.base import BaseCalibrator
from mapie.future.calibrators.utils import check_calibrator
from mapie.conformity_scores import BaseRegressionScore
from mapie.conformity_scores.interface import BaseConformityScore
from mapie.conformity_scores.utils import check_regression_conformity_score
from mapie.future.split.base import SplitCP
from mapie.utils import check_estimator_regression, check_lower_upper_bounds


class SplitCPRegressor(SplitCP):
    """
    Class to implement Conformal Prediction in ``"split"`` approach for
    regression tasks, based on :class:`~future.split.base.SplitCP`.
    It uses a predictor (``RegressorMixin`` object),
    and a calibrator (``BaseCalibrator`` object).

    Parameters
    ----------
    predictor: Optional[RegressorMixin]
        Any regressor from scikit-learn API.
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, ``predictor`` defaults to a ``LinearRegressor`` instance.

        By default ``"None"``.

    calibrator: Optional[BaseCalibrator]
        A ``BaseCalibrator`` instance used to estimate the conformity scores.

        If ``None``, use as default a ``GaussianCCP`` instance.

        By default ``None``.

    cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]]
        The splitting strategy for computing conformity scores.
        Choose among:

        - Any splitter (``ShuffleSplit`` or ``PredefinedSplit``)
        with ``n_splits=1``.
        - ``"prefit"``, assumes that ``predictor`` has been fitted already.
          All data provided in the ``calibrate`` method is then used
          for the calibration.
          The user has to take care manually that data used for model fitting
          and calibration (the data given in the ``calibrate`` method)
          are disjoint.
        - ``"split"`` or ``None``: divide the data into training and
          calibration subsets (using the default ``calib_size``=0.3).
          The splitter used is the following:
            ``sklearn.model_selection.ShuffleSplit`` with ``n_splits=1``.

        By default ``None``.

    conformity_score: Optional[BaseRegressionScore]
        BaseRegressionScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteBaseRegressionScore``
        symetrical conformity score
        - Any ``BaseRegressionScore`` class

        By default ``None``.

    alpha: Optional[float]
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``None``

    random_state: Optional[int]
        Integer used to set the numpy seed, to get reproducible calibration
        results.
        If ``None``, the prediction intervals may be stochastics and
        change if you refit the calibration (even if no arguments have change).

        .. warning::
            Some methods, as the CCP method
            (:class:`~mapie.future.calibrators.ccp.CCPCalibrator`),
            have a stochastic behavior. To have reproductible results,
            use an integer ``random_state`` value.

            However, if ``random_state`` is not ``None``, ``np.random.seed``
            will be changed, which will reset the seed for all the other random
            number generators. It may have an impact on the rest of your code.

        By default ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.future.split import SplitCPRegressor
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400, 2).reshape(-1, 1)
    >>> y_train = 2*X_train[:,0] + np.random.rand(len(X_train))
    >>> mapie_reg = SplitCPRegressor(alpha=0.1, random_state=1)
    >>> mapie_reg = mapie_reg.fit(X_train, y_train)
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    """
    def __init__(
        self,
        predictor: Optional[RegressorMixin] = None,
        calibrator: Optional[BaseCalibrator] = None,
        cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]] = None,
        alpha: Optional[float] = None,
        conformity_score: Optional[BaseRegressionScore] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.predictor = predictor
        self.conformity_score = conformity_score
        self.calibrator = calibrator
        self.alpha = alpha

    def _check_fit_parameters(self) -> RegressorMixin:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        Copy the ``predictor`` in ``predictor_`` attribute if ``cv="prefit"``.
        """
        self.cv = self._check_cv(self.cv)
        predictor = check_estimator_regression(self.predictor, self.cv)
        return predictor

    def _check_calibrate_parameters(self) -> Tuple[
        BaseRegressionScore, BaseCalibrator
    ]:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """
        conformity_score_ = check_regression_conformity_score(
            self.conformity_score, self.default_sym_
        )
        calibrator = check_calibrator(self.calibrator)
        self._check_alpha(self.alpha)
        calibrator.sym = conformity_score_.sym
        calibrator.alpha = self.alpha
        calibrator.random_state = self.random_state
        return conformity_score_, calibrator

    def get_conformity_scores(
        self,
        conformity_score: BaseConformityScore,
        X: NDArray,
        y: NDArray,
        y_pred: NDArray,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs,
    ) -> NDArray:
        """
        Return the conformity scores of the data

        Parameters
        ----------
        conformity_score: BaseRegressionScore
            Score function that handle all that is related
            to conformity scores.

        X: NDArray of shape (n_samples, n_features)
            Data

        y: NDArray of shape (n_samples,)
            Target

        y_pred: NDArray of shape (n_samples,)
            Predictions

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights of the data, used as weights in the
            calibration process.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """

        return conformity_score.get_conformity_scores(
            y, y_pred, X=X,
        )

    def predict_score(
        self, X: ArrayLike
    ) -> NDArray:
        """
        Compute the predictor prediction, used to compute the
        conformity scores.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        Returns
        -------
        NDArray of shape (n_samples, )
            predictions
        """
        return self.predictor_.predict(X)

    def predict_bounds(
        self,
        X: ArrayLike,
        y_pred: NDArray,
        **kwargs,
    ) -> NDArray:
        """
        Compute the bounds, using the fitted ``calibrator_``.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y_pred: 2D NDArray
            Observed Target

        Returns
        -------
        NDArray
            Bounds, as a 3D array of shape (n_samples, 2, 1)
            (because we only have 1 alpha value)
        """
        predict_kwargs = self._get_method_arguments(
            self.calibrator_.predict,
            dict(zip(["X", "y_pred"], [X, y_pred])),
            kwargs,
        )
        conformity_score_pred = self.calibrator_.predict(**predict_kwargs)

        self.conformity_score_ = cast(
            BaseRegressionScore, self.conformity_score_
        )
        y_pred_low = self.conformity_score_.get_estimation_distribution(
            y_pred[:, np.newaxis], conformity_score_pred[:, [0]], X=X,
        )
        y_pred_up = self.conformity_score_.get_estimation_distribution(
            y_pred[:, np.newaxis], conformity_score_pred[:, [1]], X=X,
        )

        check_lower_upper_bounds(y_pred_low, y_pred_up, y_pred)

        return np.stack([y_pred_low, y_pred_up], axis=1)

    def predict_best(self, y_pred: NDArray) -> NDArray:
        """
        Compute the prediction, in an array of shape (n_samples, )

        Parameters
        ----------
        y_pred: NDArray
            Prediction scores

        Returns
        -------
        NDArray
            Predictions
        """
        return y_pred
