import copy
from typing import Optional, Union, List, cast
from typing_extensions import Self

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator

from mapie.subsample import Subsample
from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore
from mapie.regression import MapieRegressor
from mapie.utils import check_estimator_fit_predict
from mapie_v1.conformity_scores._utils import (
    check_and_select_regression_conformity_score,
)
from mapie_v1._utils import transform_confidence_level_to_alpha_list, \
    check_if_param_in_allowed_values, check_cv_not_string, hash_X_y, \
    check_if_X_y_different_from_fit, make_intervals_single_if_single_alpha, \
    cast_point_predictions_to_ndarray


class SplitConformalRegressor:
    """
    A conformal regression model using split conformal prediction to generate
    prediction intervals.

    This method involves using a hold-out conformity set to determine
    prediction intervals around point predictions from a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals. Indicates the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level are
        returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        Valid options: see keys and values of the dictionnary
:py:const:`mapie_v1.conformity_scores.REGRESSION_CONFORMITY_SCORES_STRING_MAP`.
        See: TODO : reference conformity score classes or documentation

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    prefit : bool, default=False
        If True, assumes that the base estimator is already fitted, and skips
        refitting. If False, fits the estimator on the provided training data.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level.
        Higher values increase the output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Notes
    -----
    This implementation currently uses a ShuffleSplit cross-validation scheme
    for splitting the conformity set. Future implementations may allow the use
    of groups.

    Examples
    --------
    >>> regressor = SplitConformalRegressor(estimator=LinearRegression(),
                                            confidence_level=0.95)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        prefit: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._conformity_score = check_and_select_regression_conformity_score(
            conformity_score)

        # Note to developers: to implement this v1 class without touching the
        # v0 backend, we're for now using a hack. We always set cv="prefit",
        # and we fit the estimator if needed. See the .fit method below.
        self._mapie_regressor = MapieRegressor(
            estimator=self._estimator,
            method="base",
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self._conformity_score,
            random_state=random_state,
        )

        self._alphas = transform_confidence_level_to_alpha_list(
            confidence_level
        )

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method of the base
            estimator.

        Returns
        -------
        Self
            The fitted SplitConformalRegressor instance.
        """
        if not self._prefit:
            cloned_estimator = clone(self._estimator)
            fit_params = {} if fit_params is None else fit_params
            cloned_estimator.fit(X_train, y_train, **fit_params)
            self._mapie_regressor.estimator = cloned_estimator

        return self

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Computes conformity scores using the conformity set, allowing to
        predict intervals later on.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for the conformity set.

        y_conf : ArrayLike
            Target values for the conformity set.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The conformalized SplitConformalRegressor instance.
        """
        predict_params = {} if predict_params is None else predict_params
        self._mapie_regressor.fit(X_conf,
                                  y_conf,
                                  predict_params=predict_params)

        return self

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Generates prediction intervals for the input data `X` based on
        conformity scores and confidence level(s).

        Parameters
        ----------
        X : ArrayLike
            Data features for generating prediction intervals.

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        Returns
        -------
        NDArray
            An array containing the prediction intervals with shape
            `(n_samples, 2)` if `confidence_level` is a single float, or
            `(n_samples, 2, n_confidence_levels)` if `confidence_level` is a
            list of floats.
        """
        _, intervals = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds
        )

        return make_intervals_single_if_single_alpha(
            intervals,
            self._alphas
        )

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Generates point predictions for given data using the fitted model.

        Parameters
        ----------
        X : ArrayLike
            Data features for prediction.

        Returns
        -------
        NDArray
            Array of point predictions, with shape (n_samples,).
        """
        predictions = self._mapie_regressor.predict(X, alpha=None)
        return cast_point_predictions_to_ndarray(predictions)


class CrossConformalRegressor:
    """
    A conformal regression model using cross-conformal prediction to generate
    prediction intervals.

    This method involves computing conformity scoring across multiple folds in
    a cross-validation fashion to determine prediction intervals around point
    predictions from a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction intervals for each specified confidence level are returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        Valid options: TODO : reference here the valid options, once the list
        has been be created during the implementation.
        See: TODO : reference conformity score classes or documentation

        A custom score function inheriting from BaseRegressionScore may also be
        provided.

    method : str, default="plus"
        The method used to compute prediction intervals. Options are:
        - "base": Based on the conformity scores from each fold.
        - "plus": Based on the conformity scores from each fold and
        the test set predictions.
        - "minmax": Based on the conformity scores from each fold and
        the test set predictions, using the minimum and maximum among
        each fold models.

    cv : Union[int, BaseCrossValidator], default=5
        The cross-validation strategy used to compute confomity scores.
        Valid options:
        - integer, to specify the number of folds
        - any ``sklearn.model_selection.BaseCrossValidator`` suitable for
        regression, or a custom cross-validator inheriting from it.
        Main variants in the cross conformal setting are:
        * ``sklearn.model_selection.KFold`` (vanilla cross conformal)
        * ``sklearn.model_selection.LeaveOneOut`` (jackknife)

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the
        output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape:
        - `(n_samples, 2)` if `confidence_level` is a single float
        - `(n_samples, 2, n_confidence_levels)` if `confidence_level`
        is a list of floats.

    Examples
    --------
    >>> regressor = CrossConformalRegressor(
    ...     estimator=LinearRegression(), confidence_level=0.95, cv=10)
    >>> regressor.fit(X, y)
    >>> regressor.conformalize(X, y)
    >>> intervals = regressor.predict_set(X_test)
    """

    _VALID_METHODS = ["base", "plus", "minmax"]

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        method: str = "plus",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        check_if_param_in_allowed_values(
            method,
            "method",
            CrossConformalRegressor._VALID_METHODS
        )
        check_cv_not_string(cv)

        self._mapie_regressor = MapieRegressor(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=check_and_select_regression_conformity_score(
                conformity_score
            ),
            random_state=random_state,
        )

        self._alphas = transform_confidence_level_to_alpha_list(
            confidence_level
        )

        self._hashed_X_y: int = 0
        self._sample_weight: Optional[NDArray] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator using the entire dataset provided.

        Parameters
        ----------
        X : ArrayLike
            Features

        y : ArrayLike
            Targets

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted CrossConformalRegressor instance.
        """
        self._hashed_X_y = hash_X_y(X, y)

        if fit_params:
            fit_params_ = copy.deepcopy(fit_params)
            self._sample_weight = fit_params_.pop("sample_weight", None)
        else:
            fit_params_ = {}

        X, y, self._sample_weight, groups = self._mapie_regressor.init_fit(
            X, y, self._sample_weight, fit_params=fit_params_
        )

        self._mapie_regressor.fit_estimator(
            X, y, self._sample_weight
        )
        return self

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Computes conformity scores in a cross conformal fashion, allowing to
        predict intervals later on.

        Parameters
        ----------
        X : ArrayLike
            Features for generating conformity scores across folds.
            Must be the same X used in .fit

        y : ArrayLike
            Target values for generating conformity scores across folds.
            Must be the same y used in .fit

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/conformity set.
            By default ``None``.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The conformalized SplitConformalRegressor instance.
        """
        check_if_X_y_different_from_fit(X, y, self._hashed_X_y)
        groups = cast(Optional[NDArray], groups)
        if not predict_params:
            predict_params = {}

        self._mapie_regressor.conformalize(
            X,
            y,
            sample_weight=self._sample_weight,
            groups=groups,
            predict_params=predict_params
        )

        return self

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Generates prediction intervals for the input data `X` based on
        conformity scores and confidence level(s).

        Parameters
        ----------
        X : ArrayLike
            Data features for generating prediction intervals.

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds if
            necessary.

        Returns
        -------
        NDArray
            An array containing the prediction intervals with shape
            `(n_samples, 2)` if `confidence_level` is a single float, or
            `(n_samples, 2, n_confidence_levels)` if `confidence_level` is a
            list of floats.
        """
        # TODO: factorize this function once the v0 backend is updated with
        #  correct param names
        _, intervals = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds
        )

        return make_intervals_single_if_single_alpha(
            intervals,
            self._alphas
        )

    def predict(
        self,
        X: ArrayLike,
        aggregate_predictions: Optional[str] = None,
    ) -> NDArray:
        """
        Generates point predictions for the input data `X`:
        - using the model fitted on the entire dataset
        - or if aggregation_method is provided, aggregating predictions from
        the models fitted on each fold

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        aggregate_predictions : Optional[str], default=None
            The method to aggregate predictions across folds. Options:
            - None: No aggregation, returns predictions from the estimator
            trained on the entire dataset
            - "mean": Returns the mean prediction across folds.
            - "median": Returns the median prediction across folds.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        if not aggregate_predictions:
            ensemble = False
        else:
            ensemble = True
            self._mapie_regressor._check_agg_function(aggregate_predictions)
            self._mapie_regressor.agg_function = aggregate_predictions

        predictions = self._mapie_regressor.predict(
            X, alpha=None, ensemble=ensemble
        )
        return cast_point_predictions_to_ndarray(predictions)


class JackknifeAfterBootstrapRegressor:
    """
    A conformal regression model using the jackknife-after-bootstrap approach
    to generate prediction intervals.

    This method combines bootstrap sampling with the jackknife technique
    to produce robust prediction intervals around point predictions from
    a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level are
        returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        Valid options: TODO : reference here the valid options, once the list
        has been be created during the implementation.
        See: TODO : reference conformity score classes or documentation

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    method : str, default="plus"
        The method used for jackknife-after-bootstrap prediction. Options are:
        - "base": Based on the conformity scores from each bootstrap sample.
        - "plus": Based on the conformity scores from each bootstrap sample and
        the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores from
        each bootstrap sample.

    n_bootstraps : int, default=100
        The number of bootstrap resamples to generate for the
        jackknife-after-bootstrap procedure.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the output
        details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape
        `(n_samples, 2)`, where each row represents the lower and
        upper bounds for each sample.

    Examples
    --------
    >>> regressor = JackknifeAfterBootstrapRegressor(
    ...    estimator=LinearRegression(), confidence_level=0.9, n_bootstraps=8)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    _VALID_METHODS = ["plus", "minmax"]
    _VALID_AGGREGATION_METHODS = ["mean", "median"]

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        method: str = "plus",
        resampling: Union[int, Subsample] = 30,
        aggregation_method: str = "mean",
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        check_if_param_in_allowed_values(
            method,
            "method",
            JackknifeAfterBootstrapRegressor._VALID_METHODS
        )
        check_if_param_in_allowed_values(
            aggregation_method,
            "aggregation_method",
            JackknifeAfterBootstrapRegressor._VALID_AGGREGATION_METHODS
        )

        if isinstance(resampling, int):
            cv = Subsample(n_resamplings=resampling)
        elif isinstance(resampling, Subsample):
            cv = resampling
        else:
            raise ValueError(
                "resampling must be an integer or a Subsample instance"
            )

        self._mapie_regressor = MapieRegressor(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            agg_function=aggregation_method,
            conformity_score=check_and_select_regression_conformity_score(
                conformity_score
            ),
            random_state=random_state,
        )

        self._alphas = transform_confidence_level_to_alpha_list(
            confidence_level
        )

        self._hashed_X_y: int = 0
        self._sample_weight: Optional[NDArray] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training data features.

        y : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted JackknifeAfterBootstrapRegressor instance.
        """
        self._hashed_X_y = hash_X_y(X, y)

        if fit_params:
            fit_params_ = copy.deepcopy(fit_params)
            self._sample_weight = fit_params_.pop("sample_weight", None)
        else:
            fit_params_ = {}

        X, y, self._sample_weight, groups = self._mapie_regressor.init_fit(
            X, y, self._sample_weight, fit_params=fit_params_
        )

        self._mapie_regressor.fit_estimator(
            X, y, self._sample_weight
        )
        return self

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the model on the provided data using the
        jackknife-after-bootstrap approach, which leverages pre-generated
        bootstrap samples along with jackknife techniques to estimate
        prediction intervals. This step analyzes conformity scores and
        adjusts the intervals based on specified confidence levels.

        Parameters
        ----------
        X : ArrayLike
            Features for the calibration (conformity) data.

        y : ArrayLike
            Target values for the calibration (conformity) data.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The JackknifeAfterBootstrapRegressor instance with
            calibrated prediction intervals.
        """

        check_if_X_y_different_from_fit(X, y, self._hashed_X_y)
        if not predict_params:
            predict_params = {}

        self._mapie_regressor.conformalize(
            X,
            y,
            sample_weight=self._sample_weight,
            predict_params=predict_params
        )

        return self

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Computes prediction intervals for each sample in `X` based on
        the jackknife-after-bootstrap framework.

        Parameters
        ----------
        X : ArrayLike
            Test data for prediction intervals.

        allow_infinite_bounds : bool, default=False
            If True, allows intervals to include infinite bounds
            if required for coverage.

        Returns
        -------
        NDArray
            Prediction intervals of shape `(n_samples, 2)`,
            with lower and upper bounds for each sample.
        """
        _, intervals = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds
        )

        return make_intervals_single_if_single_alpha(
            intervals,
            self._alphas
        )

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Generates point predictions for the input data using the fitted model,
        with optional aggregation over bootstrap samples.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        predictions = self._mapie_regressor.predict(
            X, alpha=None, ensemble=True
        )
        return cast_point_predictions_to_ndarray(predictions)


class ConformalizedQuantileRegressor:
    """
    A conformal quantile regression model that generates prediction intervals
    using quantile regression as the base estimator.

    This approach provides prediction intervals by leveraging
    quantile predictions and applying conformal adjustments to ensure coverage.

    Parameters
    ----------
    estimator : RegressorMixin, default=QuantileRegressor()
        The base quantile regression estimator used to generate point and
        interval predictions.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level
        are returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        Valid options: TODO : reference here the valid options, once the list
        has been be created during the implementation.
        See: TODO : reference conformity score classes or documentation

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Methods
    -------
    fit(X_train, y_train, fit_params=None) -> Self
        Fits the base estimator to the training data and initializes internal
        parameters required for conformal prediction.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Calibrates the model on provided data, adjusting the prediction
        intervals to achieve the specified confidence levels.

    predict(X) -> NDArray
        Generates point predictions for the input data `X`.

    predict_set(X,
                allow_infinite_bounds=False,
                minimize_interval_width=False,
                symmetric_intervals=True) -> NDArray
        Generates prediction intervals for the input data `X`,
        adjusted for desired scoverage based on the calibrated
        quantile predictions.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape
        `(n_samples, 2)`,  where each row represents the lower and
        upper bounds for each sample.

    Examples
    --------
    >>> regressor = ConformalizedQuantileRegressor(
    ...     estimator=QuantileRegressor(), confidence_level=0.95)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = QuantileRegressor(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted ConformalizedQuantileRegressor instance.
        """
        return self

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the model on the provided data, adjusting the prediction
        intervals based on quantile predictions and specified confidence
        levels. This step analyzes the conformity scores and refines the
        intervals to ensure desired coverage.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for the calibration (conformity) data.

        y_conf : ArrayLike
            Target values for the calibration (conformity) data.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The ConformalizedQuantileRegressor instance with calibrated
            prediction intervals.
        """
        return self

    def predict_set(
        self,
        X: ArrayLike,
        allow_infinite_bounds: bool = False,
        minimize_interval_width: bool = False,
        symmetric_intervals: bool = True,
    ) -> NDArray:
        """
        Computes prediction intervals for quantile regression based
        on calibrated predictions.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data for prediction intervals.

        allow_infinite_bounds : bool, default=False
            If True, allows intervals to include infinite bounds
            if required for coverage.

        minimize_interval_width : bool, default=False
            If True, narrows the prediction intervals as much as possible
            while maintaining the target coverage level.

        symmetric_intervals : bool, default=True
            If True, computes symmetric intervals around the predicted
            median or mean.
            If False, calculates separate upper and lower bounds for
            asymmetric intervals.

        Returns
        -------
        NDArray
            Prediction intervals with shape `(n_samples, 2)`, with lower
            and upper bounds for each sample.
        """
        return np.ndarray(0)

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Generates point predictions for the input data using the fitted model.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        Returns
        -------
        NDArray
            Array of point predictions with shape `(n_samples,)`.
        """
        return np.ndarray(0)


class GibbsConformalRegressor:
    pass  # TODO
