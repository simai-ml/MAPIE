from typing import Optional, Union, List
from typing_extensions import Self

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore
from mapie.regression import MapieRegressor
from mapie.utils import check_estimator_fit_predict
from mapie_v1.conformity_scores.utils import \
    check_and_select_split_conformity_score


class SplitConformalRegressor:
    """
    A conformal regression model using split conformal prediction to generate
    prediction intervals with statistical guarantees. This method involves
    using a hold-out conformity set to determine prediction intervals around
    point predictions from a base regressor.

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
        self.estimator = estimator
        self.prefit = prefit
        self.conformity_score = check_and_select_split_conformity_score(
            conformity_score)

        # Note to developers: to implement this v1 class without touching the
        # v0 backend, we're for now using a hack. We always set cv="prefit",
        # and we fit the estimator if needed. See the .fit method below.
        self.mapie_regressor = MapieRegressor(
            estimator=self.estimator,
            method="base",
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self.conformity_score,
            random_state=random_state,
        )

        if isinstance(confidence_level, float):
            confidence_level = [confidence_level]

        self.alpha = [1 - level for level in confidence_level]

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
        if not self.prefit:
            cloned_estimator = clone(self.estimator)
            fit_params = {} if fit_params is None else fit_params
            cloned_estimator.fit(X_train, y_train, **fit_params)
            self.mapie_regressor.estimator = cloned_estimator

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
            The SplitConformalRegressor instance with updated prediction
            intervals.
        """
        predict_params = {} if predict_params is None else predict_params
        self.mapie_regressor.fit(X_conf,
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
        _, intervals = self.mapie_regressor.predict(
            X,
            alpha=self.alpha,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds
        )

        if len(self.alpha) == 1:
            intervals = intervals[:, :, 0]

        return intervals

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
        predictions = self.mapie_regressor.predict(X, alpha=None)
        return predictions


class CrossConformalRegressor:
    """
    A conformal regression model using cross-conformal prediction to generate
    prediction intervals with statistical guarantees. This method involves
    cross-validation with conformity scoring across multiple folds to determine
    prediction intervals around point predictions from a base regressor.

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
        The method used for cross-conformal prediction. Options are:
        - "base": Based on the conformity scores from each fold.
        - "plus": Based on the conformity scores from each fold plus
          the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores
          from each fold.

    cv : Union[int, BaseCrossValidator], default=5
        The cross-validation splitting strategy. If an integer is passed, it is
        the number of folds for `KFold` cross-validation. Alternatively, a
        specific cross-validation splitter from scikit-learn can be provided.
        TODO : reference here the valid options,
         once the list has been be created during the implementation

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the
        output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Methods
    -------
    fit(X_train, y_train, fit_params=None) -> Self
        Fits the base estimator to the training data using cross-validation.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Calibrates the model using cross-validation and updates the prediction
        intervals based on conformity errors observed across folds.

    predict(X, aggregation_method=None) -> NDArray
        Generates point predictions for the input data `X` using the specified
        aggregation method across the cross-validation folds.

    predict_set(X, minimize_interval_width=False, allow_infinite_bounds=False)
    -> NDArray
        Generates prediction intervals for the input data `X` based on the
        conformity score and confidence level, adjusted to achieve the desired
        coverage probability.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape:
        - `(n_samples, 2)` if `confidence_level` is a single float
        - `(n_samples, 2, n_confidence_levels)` if `confidence_level`
           is a list of floats.

    Notes
    -----
    Cross-conformal prediction provides enhanced robustness through the
    aggregation of multiple conformal scores across cross-validation folds,
    potentially yielding tighter intervals with reliable coverage guarantees.

    Examples
    --------
    >>> regressor = CrossConformalRegressor(
    ...     estimator=LinearRegression(), confidence_level=0.95, cv=10)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

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
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data using cross-validation.

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
            The fitted CrossConformalRegressor instance.
        """
        pass

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the fitted model using cross-validation conformal folds.
        This step analyzes conformity scores across multiple cross-validation
        folds and adjusts the prediction intervals based on conformity errors
        and specified confidence levels.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for generating conformity scores across folds.

        y_conf : ArrayLike
            Target values for generating conformity scores across folds.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The CrossConformalRegressor instance with calibrated prediction
            intervals based on cross-validated conformity scores.
        """
        pass

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Generates prediction intervals for the input data `X` based on the
        calibrated model and cross-conformal prediction framework.

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
            Prediction intervals with shape
            - `(n_samples, 2)` if `confidence_level`is a single float,
            - `(n_samples, 2, n_confidence_levels)` if multiple confidence
               levels are specified.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        aggregation_method: Optional[str] = None,
    ) -> NDArray:
        """
        Generates point predictions for the input data `X` using the
        fitted model. Optionally aggregates predictions across cross-
        validation folds models.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        aggregation_method : Optional[str], default=None
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
        pass


class JackknifeAfterBootstrapRegressor:
    """
    A conformal regression model using the jackknife-after-bootstrap approach
    to generate prediction intervals with statistical guarantees. This method
    combines bootstrap sampling with the jackknife technique to produce robust
    prediction intervals around point predictions from a base regressor.

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

    Methods
    -------
    fit(X_train, y_train, fit_params=None) -> Self
        Fits the base estimator to the training data and initializes internal
        parameters required for the jackknife-after-bootstrap process.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Calibrates the model on provided data using the
        jackknife-after-bootstrap approach, adjusting the prediction intervals
        based on the observed conformity scores.

    predict(X, aggregation_method="mean") -> NDArray
        Generates point predictions for the input data `X` using the specified
        aggregation method over bootstrap samples.

    predict_set(X, allow_infinite_bounds=False) -> NDArray
        Generates prediction intervals for the input data `X` based on the
        calibrated jackknife-after-bootstrap predictions.

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

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        method: str = "plus",
        n_bootstraps: int = 100,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
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
            The fitted JackknifeAfterBootstrapRegressor instance.
        """

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
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
            The JackknifeAfterBootstrapRegressor instance with
            calibrated prediction intervals.
        """

    def predict_set(
        self,
        X: ArrayLike,
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
        pass

    def predict(
        self,
        X: ArrayLike,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: str = "mean",  # 'mean', 'median'
    ) -> NDArray:
        """
        Generates point predictions for the input data using the fitted model,
        with optional aggregation over bootstrap samples.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        aggregation_method : str, default="mean"
            The method to aggregate predictions across bootstrap samples.
            Options:
            - "mean": Returns the mean prediction across samples.
            - "median": Returns the median prediction across samples.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        pass


class ConformalizedQuantileRegressor:
    """
    A conformal quantile regression model that generates prediction intervals
    with statistical guarantees using quantile regression as the base
    estimator. This approach provides prediction intervals by leveraging
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
        # n_jobs: Optional[int] = None
        # Not yet available in MapieQuantileRegressor
        # verbose: int = 0,
        # Idem
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
        pass

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
        pass


class GibbsConformalRegressor:
    pass  # TODO
