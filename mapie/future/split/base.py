from __future__ import annotations

import inspect
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (BaseCrossValidator,
                                     PredefinedSplit, ShuffleSplit)
from sklearn.utils.validation import _num_samples, check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.future.calibrators.base import BaseCalibrator
from mapie.conformity_scores.interface import BaseConformityScore
from mapie.utils import _sample_non_null_weight, fit_estimator


class SplitCP(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for Split Conformal Prediction

    Parameters
    ----------
    predictor: Optional[BaseEstimator]
        Any estimator from scikit-learn API.
        (i.e. with ``fit`` and ``predict`` methods).

        If ``None``, will default to a value defined by the subclass

        By default ``None``.

    calibrator: Optional[BaseCalibrator]
        A ``BaseCalibrator`` instance used to estimate the conformity scores.

        If ``None``, defaults to a ``GaussianCCP`` calibrator.

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

    conformity_score: Optional[BaseConformityScore]
        ``BaseConformityScore`` instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores.

        - Can be any ``BaseConformityScore`` class
        - ``None`` is associated with a default value defined by the subclass

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
        If ``None``, the prediction intervals will be stochastics, and will
        change if you refit the calibration (even if no arguments have change).

        WARNING: If ``random_state`` is not ``None``, ``np.random.seed`` will
        be changed, which will reset the seed for all the other random
        number generators. It may have an impact on the rest of your code.

        By default ``None``.
    """

    default_sym_ = True
    fit_attributes = ["predictor_"]
    calib_attributes = ["calibrator_"]

    cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]]
    alpha: Optional[float]

    @abstractmethod
    def __init__(
        self,
        predictor: Optional[BaseEstimator] = None,
        calibrator: Optional[BaseCalibrator] = None,
        cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]] = None,
        alpha: Optional[float] = None,
        conformity_score: Optional[BaseConformityScore] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialisation
        """

    @abstractmethod
    def _check_fit_parameters(self) -> BaseEstimator:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        """

    @abstractmethod
    def _check_calibrate_parameters(self) -> Tuple[
        BaseConformityScore, BaseCalibrator
    ]:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """

    def _check_cv(
        self,
        cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]] = None,
        test_size: Optional[Union[int, float]] = None,
    ) -> Union[str, ShuffleSplit, PredefinedSplit]:
        """
        Check if ``cv`` is ``None``, ``"prefit"``, ``"split"``,
        or ``ShuffleSplit``/``PredefinedSplit`` with ``n_splits=1``.

        Return a ``ShuffleSplit`` instance with ``n_splits=1``
        if ``None`` or ``"split"``.

        Else raise error.

        Parameters
        ----------
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]]
            Cross-validator to check, by default ``None``.

        test_size: float
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split.
            If cv is not ``"split"``, ``test_size`` is ignored.

            By default ``None``.

        Returns
        -------
        Union[str, PredefinedSplit, ShuffleSplit]
            The cast `cv` parameter.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None or cv == "split":
            return ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.random_state
            )
        elif (isinstance(cv, (PredefinedSplit, ShuffleSplit))
              and cv.get_n_splits() == 1):
            return cv
        elif cv == "prefit":
            return cv
        else:
            raise ValueError(
                "Invalid cv argument.  Allowed values are None, 'prefit', "
                "'split' or a `ShuffleSplit/PredefinedSplit` object with "
                "`n_splits=1`."
            )

    def _check_alpha(self, alpha: Optional[float] = None) -> None:
        """
        Check the ``alpha`` parameter.

        Parameters
        ----------
        alpha: Optional[float]
            Can be a float between 0 and 1, represent the uncertainty
            of the confidence interval. Lower alpha produce
            larger (more conservative) prediction intervals.
            alpha is the complement of the target coverage level.

        Raises
        ------
        ValueError
            If alpha is not ``None`` or a float between 0 and 1.
        """
        if alpha is None:
            return
        if isinstance(alpha, float):
            alpha = alpha
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )

        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha. "
                             "Allowed values are between 0 and 1.")

    def _get_method_arguments(
        self, method: Callable, local_vars: Dict[str, Any],
        kwargs: Optional[Dict],
    ) -> Dict:
        """
        Return a dictionnary of the ``method`` arguments.

        The arguments of ``method`` must be attributes of ``self``, in
        ``local_vars``, or in ``kwargs``.

        Parameters
        ----------
        method: Callable
            method for which to check the signature

        local_vars : Dict[str, Any]
            Dictionnary of available variables

        kwargs : Optional[Dict]
            Other arguments

        exclude_args : Optional[List[str]]
            Arguments to exclude

        Returns
        -------
        Dict
            dictinnary of arguments
        """
        self_attrs = {k: v for k, v in self.__dict__.items()}
        sig = inspect.signature(method)

        method_kwargs: Dict[str, Any] = {}
        for param in sig.parameters.values():
            # We ignore the arguments like *args and **kwargs of the method
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY):
                param_name = param.name
                if kwargs is not None and param_name in kwargs:
                    method_kwargs[param_name] = kwargs[param_name]
                elif param_name in self_attrs:
                    method_kwargs[param_name] = self_attrs[param_name]
                elif param_name in local_vars:
                    method_kwargs[param_name] = local_vars[param_name]

        return method_kwargs

    def _check_conformity_scores(self, conformity_scores: NDArray) -> NDArray:
        """
        Check the conformity scores shape

        Parameters
        ----------
        conformity_scores : NDArray of shape (n_samples,) or (n_sampels, 1)
            Conformity scores

        Returns
        -------
        NDArray:
            Conformity scores as 1D-array of shape (n_samples,)
        """
        if len(conformity_scores.shape) == 1:
            return conformity_scores
        if conformity_scores.shape[1] == 1:
            return conformity_scores[:, 0]
        else:
            raise ValueError(
                "Invalid conformity scores. The `get_conformity_scores`"
                "method of the calibrator, should return an array of shape"
                "(n_samples,) or (n_samples, 1)."
                f"Got {conformity_scores.shape}."
            )

    def fit_predictor(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_kwargs,
    ) -> SplitCP:
        """
        Fit the predictor if ``cv`` argument is not ``"prefit"``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights used in the predictor fitting.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples, used while splitting the dataset into
            train/test set.

            By default ``None``.

        **fit_kwargs: dict
            Additional fit parameters for the predictor.

        Returns
        -------
        SplitCP
            self
        """
        predictor = self._check_fit_parameters()

        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            train_index, _ = list(self.cv.split(X, y, groups))[0]

            (
                X_train, y_train, _, sample_weight_train, _
            ) = _sample_non_null_weight(X, y, sample_weight, train_index)

            self.predictor_ = fit_estimator(
                predictor, X_train, y_train,
                sample_weight=sample_weight_train, **fit_kwargs
            )
        else:
            self.predictor_ = predictor
        return self

    def fit_calibrator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **calib_kwargs,
    ) -> SplitCP:
        """
        Fit the calibrator. Arguments of the calibrator's ``fit`` method
        that are not in the following list:
        ``X, y, z, sample_weight, groups, y_pred_calib,
        conformity_scores_calib,
        X_train, y_train, z_train, sample_weight_train, train_index,
        X_calib, y_calib, z_calib, sample_weight_calib, calib_index``
        nor attributes of the ``SplitCP`` instance,
        must be given by the user in ``**calib_kwargs``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Data

        y: ArrayLike of shape (n_samples,)
            Target

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights of the data, used as weights in the
            calibration process.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        calib_kwargs: dict
            Additional fit parameters for the calibrator, used as kwargs.
            See the calibrator ``.fit`` method documentation to have more
            information about the required arguments.

            .. note::
                if the calibrator need exogenous variables (``z_train`` or
                ``z_calib``), you should pass ``z`` in ``calib_kwargs``

        Returns
        -------
        SplitCP
            self
        """
        self._check_fit_parameters()
        self.conformity_score_, calibrator = self._check_calibrate_parameters()
        check_is_fitted(self, self.fit_attributes)

        if self.alpha is None:
            warnings.warn("No calibration is done, because alpha is None.")
            return self

        # Get training and calibration sets
        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            train_index, calib_index = list(self.cv.split(X, y, groups))[0]
        else:
            train_index, calib_index = (np.array([], dtype=int),
                                        np.arange(_num_samples(X)))

        z = cast(Optional[ArrayLike], calib_kwargs.get("z", None))
        (
            X_train, y_train, z_train, sample_weight_train, train_index
        ) = _sample_non_null_weight(X, y, sample_weight, train_index, z)
        (
            X_calib, y_calib, z_calib, sample_weight_calib, calib_index
        ) = _sample_non_null_weight(X, y, sample_weight, calib_index, z)

        # Compute conformity scores
        y_pred_calib = self.predict_score(X_calib)

        y_calib = cast(NDArray, y_calib)
        X_calib = cast(NDArray, X_calib)

        conformity_scores_calib = self.get_conformity_scores(
            self.conformity_score_, X_calib, y_calib,
            y_pred_calib, sample_weight_calib, groups
        )

        conformity_scores_calib = self._check_conformity_scores(
            conformity_scores_calib
        )

        # Get the calibrator arguments
        dict_arguments = dict(zip([
            "X", "y", "z", "sample_weight", "groups",
            "y_pred_calib", "conformity_scores_calib",
            "X_train", "y_train", "z_train",
            "sample_weight_train", "train_index",
            "X_calib", "y_calib", "z_calib",
            "sample_weight_calib", "calib_index",
            ],
            [
            X, y, z, sample_weight, groups,
            y_pred_calib, conformity_scores_calib,
            X_train, y_train, z_train, sample_weight_train, train_index,
            X_calib, y_calib, z_calib, sample_weight_calib, calib_index,
        ]))
        calib_arguments = self._get_method_arguments(
            calibrator.fit,
            dict_arguments,
            calib_kwargs
        )

        self.calibrator_ = calibrator.fit(
            **calib_arguments,
            **(
                {
                    key: calib_kwargs[key] for key in calib_kwargs
                    if key not in dict_arguments
                }
                if calib_kwargs is not None
                else {}
            )
        )

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        fit_kwargs: Optional[Dict] = None,
        calib_kwargs: Optional[Dict] = None
    ) -> SplitCP:
        """
        Fit the predictor (if ``cv`` is not ``"prefit"``)
        and fit the calibrator.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Data

        y: ArrayLike of shape (n_samples,)
            Target

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights used in the predictor fitting.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        fit_kwargs: dict
            Additional fit parameters for the predictor, used as kwargs.

        calib_kwargs: dict
            Additional fit parameters for the calibrator, used as kwargs.
            See the calibrator ``.fit`` method documentation to have more
            information about the required arguments.

            .. note::
                if the calibrator need exogenous variables (``z_train`` or
                ``z_calib``), you should pass ``z`` in ``calib_kwargs``

        Returns
        -------
        SplitCP
            self
        """
        self.fit_predictor(X, y, sample_weight, groups,
                           **(fit_kwargs if fit_kwargs is not None else {}))
        self.fit_calibrator(X, y, sample_weight, groups,
                            **(calib_kwargs
                               if calib_kwargs is not None else {}))
        return self

    def predict(
        self,
        X: ArrayLike,
        **kwargs,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with prediction intervals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        kwargs: dict
            Additional predict parameters for the calibrator, used as kwargs.
            See the calibrator ``.predict`` method documentation to have more
            information about the required arguments.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - Predictions : NDArray of shape (n_samples,)
            if ``alpha`` is ``None``.
            - Prediction intervals
            if ``alpha`` is not ``None``.
        """
        check_is_fitted(self, self.fit_attributes)
        y_pred = self.predict_score(X)

        if self.alpha is None:
            return self.predict_best(y_pred)

        check_is_fitted(self, self.calib_attributes)

        # Fit the calibrator
        bounds_arguments = self._get_method_arguments(
            self.calibrator_.predict, {}, kwargs,
        )

        y_bounds = self.predict_bounds(X, y_pred, **bounds_arguments)

        return self.predict_best(y_pred), y_bounds

    @abstractmethod
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

    @abstractmethod
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
        NDArray
            Scores (usually ``y_pred`` in regression and ``y_pred_proba``
            in classification)
        """

    @abstractmethod
    def predict_bounds(
        self,
        X: ArrayLike,
        y_pred: NDArray,
        **predict_kwargs,
    ) -> NDArray:
        """
        Compute the bounds, using the fitted ``calibrator_``.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y_pred: 2D NDArray
            Predicted scores (target)

        z: Optional[ArrayLike]
            Exogenous variables

        Returns
        -------
        NDArray
            Bounds (or prediction set in classification)
        """

    @abstractmethod
    def predict_best(self, y_pred: NDArray) -> NDArray:
        """
        Compute the best prediction, in an array of shape (n_samples, )

        Parameters
        ----------
        y_pred: NDArray
            Prediction scores (can be the prediction, the probas, ...)

        z: Optional[ArrayLike]
            Exogenous variables

        Returns
        -------
        NDArray
            predictions
        """
