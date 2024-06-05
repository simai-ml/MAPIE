from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import (BaseCrossValidator, BaseShuffleSplit,
                                     ShuffleSplit, PredefinedSplit)
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import ConformityScore
from .utils.ccp_phi_function import PhiFunction
from mapie.utils import (check_conformity_score, check_estimator,
                         check_lower_upper_bounds, check_null_weight,
                         fit_estimator)


class MapieCCPRegressor(BaseEstimator, RegressorMixin):
    """
    This class implements Conformal Prediction With Conditional Guarantees
    method as proposed by Gibbs et al. (2023) to make conformal predictions.
    This method works with a ``"split"`` approach which requires a separate
    calibration phase. The ``calibrate`` method is used on a calibration set
    that must be disjoint from the estimator's training set to guarantee
    the expected ``1-alpha`` coverage.

    Parameters
    ----------
    estimator: Optional[RegressorMixin]
        Any regressor from scikit-learn API.
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, ``estimator`` defaults to a ``LinearRegressor`` instance.

        By default ``"None"``.

    phi: Optional[PhiFunction]
        A ``PhiFunction`` instance used to estimate the conformity scores.

        If ``None``, use as default a ``GaussianPhiFunction`` instance.
        See the examples and the documentation to build a ``PhiFunction``
        adaptated to your dataset and constraints.

        By default ``None``.

    cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]]
        The splitting strategy for computing conformity scores.
        Choose among:

        - Any splitter (``ShuffleSplit`` or ``PredefinedSplit``)
        with ``n_splits=1``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
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

    conformity_score: Optional[ConformityScore]
        ConformityScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` symetrical
        conformity score
        - Any ``ConformityScore`` class

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

        WARNING: If ``random_state``is not ``None``, ``np.random.seed`` will
        be changed, which will reset the seed for all the other random
        number generators. It may have an impact on the rest of your code.

        By default ``None``.

    Attributes
    ----------
    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up[0]: Array of shape (phi.n_out, )
        beta_up[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low_: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieCCPRegressor
    >>> X_train = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_train = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> mapie_reg = MapieCCPRegressor(alpha=0.1, random_state=1)
    >>> mapie_reg = mapie_reg.fit_calibrate(
    ...     X_train,
    ...     y_train,
    ... )
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(np.round(y_pis[:,:, 0], 2))
    [[ 4.14  5.57]
     [ 6.11  7.54]
     [ 8.07  9.5 ]
     [10.04 11.46]
     [12.   13.43]
     [13.96 15.39]]
    >>> print(np.round(y_pred, 2))
    [ 4.86  6.82  8.79 10.75 12.71 14.68]
    """

    default_sym_ = True
    fit_attributes = ["estimator_"]
    calib_attributes = ["beta_up_", "beta_low_"]

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        phi: Optional[PhiFunction] = None,
        cv: Optional[
            Union[str, BaseCrossValidator, BaseShuffleSplit]
        ] = None,
        alpha: Optional[float] = None,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.estimator = estimator
        self.conformity_score = conformity_score
        self.phi = phi
        self.alpha = alpha

    def _check_parameters(self) -> None:
        """
        Check and replace default value of ``estimator`` and ``cv`` arguments.
        Copy the ``estimator`` in ``estimator_`` attribute if ``cv="prefit"``.
        """
        self.cv = self._check_cv(self.cv)
        self.estimator = check_estimator(self.estimator, self.cv)

        if self.cv == "prefit":
            self.estimator_ = self.estimator

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike],
        train_index: ArrayLike,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """
        Perform several checks on class parameters.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        sample_weight: Optional[NDArray] of shape (n_samples,)
            Non-null sample weights.

        train_index: ArrayLike
            Indexes of the training set.

        Returns
        -------
        Tuple[NDArray, NDArray, Optional[NDArray]]
            - NDArray of training observed values
            - NDArray of training target values
            - Optional[NDArray] of training sample_weight
        """
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)

        if sample_weight is not None:
            sample_weight_train = _safe_indexing(
                sample_weight, train_index)
        else:
            sample_weight_train = None

        X_train, y_train = indexable(X_train, y_train)
        y_train = _check_y(y_train)
        sample_weight_train, X_train, y_train = check_null_weight(
            sample_weight_train, X_train, y_train)

        X_train = cast(NDArray, X_train)
        y_train = cast(NDArray, y_train)
        sample_weight_train = cast(Optional[NDArray], sample_weight_train)

        return X_train, y_train, sample_weight_train

    def _check_calibrate_parameters(self) -> None:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``phi`` arguments.
        """
        self.conformity_score_ = check_conformity_score(
            self.conformity_score, self.default_sym_
        )
        self.alpha = self._check_alpha(self.alpha)
        self.phi = self._check_phi(self.phi)

    def _check_phi(
        self,
        phi: Optional[PhiFunction],
    ) -> PhiFunction:
        """
        Check if ``phi`` is a ``PhiFunction`` instance.

        Parameters
        ----------
        phi: Optional[PhiFunction]
            A ``PhiFunction`` instance used to estimate the conformity scores.

            If ``None``, use as default a ``GaussianPhiFunction`` instance.
            See the examples and the documentation to build a ``PhiFunction``
            adaptated to your dataset and constraints.

        Returns
        -------
        PhiFunction
            ``phi`` if defined, a ``GaussianPhiFunction`` instance otherwise.

        Raises
        ------
        ValueError
            If ``phi`` is not ``None`` nor a ``PhiFunction`` instance.
        """
        if phi is None:
            return GaussianPhiFunction()
        elif isinstance(phi, PhiFunction):
            return phi
        else:
            raise ValueError("Invalid `phi` argument. It must be `None` or a "
                             "`PhiFunction` instance.")

    def _check_cv(
        self,
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]] = None,
        test_size: float = 0.3,
    ) -> Union[str, BaseCrossValidator, BaseShuffleSplit]:
        """
        Check if ``cv`` is ``None``, ``"prefit"``, ``"split"``,
        or ``BaseShuffleSplit``/``BaseCrossValidator`` with ``n_splits``=1.
        Return a ``ShuffleSplit`` instance ``n_splits``=1
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
                "'split' or a ShuffleSplit/PredefinedSplit object with "
                "``n_splits=1``."
            )

    def _check_alpha(
        self,
        alpha: Optional[float] = None
    ) -> Optional[float]:
        """
        Check alpha

        Parameters
        ----------
        alpha: Optional[float]
            Can be a float between 0 and 1, represent the uncertainty
            of the confidence interval. Lower alpha produce
            larger (more conservative) prediction intervals.
            alpha is the complement of the target coverage level.

        Returns
        -------
        Optional[float]
            Valid alpha.

        Raises
        ------
        ValueError
            If alpha is not ``None`` or a float between 0 and 1.
        """
        if alpha is None:
            return alpha
        if isinstance(alpha, float):
            alpha = alpha
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )

        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha. "
                             "Allowed values are between 0 and 1.")
        return alpha

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> MapieCCPRegressor:
        """
        Fit the estimator if ``cv`` argument is not ``"prefit"``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        **fit_params: dict
            Additional fit parameters for the estimator.

        Returns
        -------
        MapieCCPRegressor
            self
        """
        self._check_parameters()

        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            train_index, _ = list(self.cv.split(X, y, groups))[0]

            (
                X_train, y_train, sample_weight_train
            ) = self._check_fit_parameters(X, y, sample_weight, train_index)

            self.estimator_ = fit_estimator(
                self.estimator, X_train, y_train,
                sample_weight=sample_weight_train, **fit_params
            )
        return self

    def calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
    ) -> MapieCCPRegressor:
        """
        Calibrate with (``X``, ``y`` and ``z``)
        and the new value ``alpha`` value, if not ``None``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``

        alpha: Optional[float]
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            If ``None``, the calibration will be done using the ``alpha``value
            set in the initialisation. Else, the new value will overwrite the
            old one.

            By default ``None``

        Returns
        -------
        MapieCCPRegressor
            self
        """
        self._check_parameters()
        self._check_calibrate_parameters()
        check_is_fitted(self, self.fit_attributes)
        self.phi = cast(PhiFunction, self.phi)

        self.estimator = cast(RegressorMixin, self.estimator)
        self.cv = cast(Union[str, BaseCrossValidator], self.cv)
        self.conformity_score_ = cast(ConformityScore, self.conformity_score_)

        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            _, calib_index = list(self.cv.split(X, y, groups))[0]
            X_calib = _safe_indexing(X, calib_index)
            y_calib = _safe_indexing(y, calib_index)
            if z is not None:
                z_calib = _safe_indexing(z, calib_index)
            else:
                z_calib = None
        else:
            X_calib, y_calib, z_calib = X, y, z

        if alpha is not None and self.alpha != alpha:
            self.alpha = self._check_alpha(alpha)
            warnings.warn(f"WARNING: The old value of alpha ({self.alpha}) "
                          f"has been overwritten by the new one ({alpha}).")

        if self.alpha is None:
            return self

        y_pred_calib = self.estimator_.predict(X_calib)

        calib_conformity_scores = self.conformity_score_.get_conformity_scores(
            X_calib, y_calib, y_pred_calib
        )

        if self.conformity_score_.sym:
            alpha_low = 1 - self.alpha
            alpha_up = 1 - self.alpha
        else:
            alpha_low = self.alpha / 2
            alpha_up = 1 - self.alpha / 2

        def l_alpha(alpha, X, S):
            return np.where(S >= X, (1 - alpha) * (S - X), alpha * (X - S))

        def sum_of_losses(beta, phi_x, S, alpha):
            return np.sum(l_alpha(alpha, phi_x.dot(beta), S))

        phi_x = self.phi(
            X_calib,
            cast(NDArray, y_pred_calib),
            cast(NDArray, z_calib),
        )

        if np.any(np.all(phi_x == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "phi(X, y_pred, z) is full of zeros. "
                          "It will result in a prediction interval of zero "
                          "width. Consider changing the PhiFunction "
                          "definintion.\n"
                          "Fix: Use `marginal_guarantee`=True in PhiFunction")

        not_nan_index = np.where(~np.isnan(calib_conformity_scores))[0]
        # Some conf. score values may be nan (ex: with ResidualNormalisedScore)

        if self.random_state is None:
            warnings.warn("WARNING: The method implemented in "
                          "MapieCCPRegressor has a stochastic behavior. "
                          "To have reproductible results, use a integer "
                          "`random_state` value in the `MapieCCPRegressor` "
                          "initialisation.")
        else:
            np.random.seed(self.random_state)

        optimal_beta_up = minimize(
            sum_of_losses, np.random.normal(0, 1, self.phi.n_out),
            args=(
                phi_x[not_nan_index, :],
                calib_conformity_scores[not_nan_index],
                1-alpha_up
                )
            )

        if not self.conformity_score_.sym:
            optimal_beta_low = minimize(
                sum_of_losses, np.random.normal(0, 1, self.phi.n_out),
                args=(
                    phi_x[not_nan_index, :],
                    calib_conformity_scores[not_nan_index],
                    1-alpha_low
                )
            )
        else:
            optimal_beta_low = optimal_beta_up

        if not optimal_beta_up.success:
            warnings.warn(
                "WARNING: The optimization process for the upper bound "
                f"failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )
        if (not self.conformity_score_.sym
           and not optimal_beta_low.success):
            warnings.warn(
                "WARNING: The optimization process for the lower bound "
                f"failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )

        self.beta_up_ = cast(Tuple[NDArray, bool],
                             (optimal_beta_up.x, optimal_beta_up.success))
        self.beta_low_ = cast(Tuple[NDArray, bool],
                              (optimal_beta_low.x, optimal_beta_low.success))
        return self

    def fit_calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        **fit_params,
    ) -> MapieCCPRegressor:
        """
        Fit the estimator (if ``cv`` is not ``"prefit"``)
        and fit the calibration.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``

        alpha: Optional[float]
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            If ``None``, the calibration will be done using the ``alpha``value
            set in the initialisation. Else, the new value will overwrite the
            old one.

            By default ``None``

        **fit_params: dict
            Additional fit parameters for the estimator.

        Returns
        -------
        MapieCCPRegressor
            self
        """
        self.fit(X, y, sample_weight, groups, **fit_params)
        self.calibrate(X, y, groups, z, alpha)
        return self

    def predict(
        self,
        X: ArrayLike,
        z: Optional[ArrayLike] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        The prediction interval is computed

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
                - [:, 0, :]: Lower bound of the prediction interval.
                - [:, 1, :]: Upper bound of the prediction interval.
        """
        check_is_fitted(self, self.fit_attributes)
        y_pred = self.estimator_.predict(X)

        if self.alpha is None:
            return y_pred

        check_is_fitted(self, self.calib_attributes)

        signed = -1 if self.conformity_score_.sym else 1

        y_pred_low = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis],
            phi_x.dot(signed * self.beta_low_[0][:, np.newaxis])
        )
        y_pred_up = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis],
            phi_x.dot(self.beta_up_[0][:, np.newaxis])
        )

        check_lower_upper_bounds(y_pred_low, y_pred_up, y_pred)

        return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
