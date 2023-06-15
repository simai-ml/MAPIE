from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from mapie._compatibility import np_nanquantile
from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all, phi2D
from mapie.conformity_scores import ConformityScore
from mapie.utils import (check_alpha, check_alpha_and_n_samples,
                         check_conformity_score, check_cv,
                         check_estimator_fit_predict, check_n_features_in,
                         check_n_jobs, check_nan_in_aposteriori_prediction,
                         check_null_weight, check_verbose, fit_estimator)


class MapieRegressor(BaseEstimator, RegressorMixin):
    """
    Prediction interval with out-of-fold conformity scores.

    This class implements the jackknife+ strategy and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold conformity scores (signed residuals,
    absolute residuals, residuals normalized by the predicted mean...)
    on hold-out validation sets and to deduce valid confidence intervals
    with strong theoretical guarantees.

    Parameters
    ----------
    estimator: Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

        By default ``None``.

    method: str
        Method to choose for prediction interval estimates.
        Choose among:

        - ``"naive"``, based on training set conformity scores,
        - ``"base"``, based on validation sets conformity scores,
        - ``"plus"``, based on validation conformity scores and
          testing predictions,
        - ``"minmax"``, based on validation conformity scores and
          testing predictions (min/max among cross-validation clones).

        By default ``"plus"``.

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing conformity scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to ``-1``, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
            - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
            - ``sklearn.model_selection.KFold`` (cross-validation),
            - ``subsample.Subsample`` object (bootstrap).
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is ignored.
          All data provided in the ``fit`` method is then used
          for computing conformity scores only.
          At prediction time, quantiles of these conformity scores are used
          to provide a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          conformity scores estimate are disjoint.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If ``float``, should be between ``0.0`` and ``1.0`` and represent the
        proportion of the dataset to include in the test split. If ``int``,
        represents the absolute number of test samples. If ``None``,
        it will be set to ``0.1``.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For ``n_jobs`` below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        ``None`` is a marker for `unset` that will be interpreted as
        ``n_jobs=1`` (sequential execution).

        By default ``None``.

    agg_function: Optional[str]
        Determines how to aggregate predictions from perturbed models, both at
        training and prediction time.

        If ``None``, it is ignored except if ``cv`` class is ``Subsample``,
        in which case an error is raised.
        If ``"mean"`` or ``"median"``, returns the mean or median of the
        predictions computed from the out-of-folds models.
        Note: if you plan to set the ``ensemble`` argument to ``True`` in the
        ``predict`` method, you have to specify an aggregation function.
        Otherwise an error would be raised.

        The Jackknife+ interval can be interpreted as an interval around the
        median prediction, and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        When the cross-validation strategy is ``Subsample`` (i.e. for the
        Jackknife+-after-Bootstrap method), this function is also used to
        aggregate the training set in-sample predictions.

        If ``cv`` is ``"prefit"`` or ``"split"``, ``agg_function`` is ignored.

        By default ``"mean"``.

    verbose: int
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    conformity_score: Optional[ConformityScore]
        ConformityScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` conformity
          score
        - ConformityScore: any ``ConformityScore`` class

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_: sklearn.RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: list
        List of out-of-folds estimators.

    conformity_scores_: ArrayLike of shape (n_samples_train,)
        Conformity scores between ``y_train`` and ``y_pred``.

    k_: ArrayLike
        - Array of nans, of shape (len(y), 1) if ``cv`` is ``"prefit"``
          (defined but not used)
        - Dummy array of folds containing each training sample, otherwise.
          Of shape (n_samples_train, cv.get_n_splits(X_train, y_train)).

    n_features_in_: int
        Number of features passed to the ``fit`` method.

    References
    ----------
    Rina Foygel Barber, Emmanuel J. Candès,
    Aaditya Ramdas, and Ryan J. Tibshirani.
    "Predictive inference with the jackknife+."
    Ann. Statist., 49(1):486-507, February 2021.

    Byol Kim, Chen Xu, and Rina Foygel Barber.
    "Predictive Inference Is Free with the Jackknife+-after-Bootstrap."
    34th Conference on Neural Information Processing Systems (NeurIPS 2020).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> X_toy = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> clf = LinearRegression().fit(X_toy, y_toy)
    >>> mapie_reg = MapieRegressor(estimator=clf, cv="prefit")
    >>> mapie_reg = mapie_reg.fit(X_toy, y_toy)
    >>> y_pred, y_pis = mapie_reg.predict(X_toy, alpha=0.5)
    >>> print(y_pis[:, :, 0])
    [[ 4.95714286  5.61428571]
     [ 6.84285714  7.5       ]
     [ 8.72857143  9.38571429]
     [10.61428571 11.27142857]
     [12.5        13.15714286]
     [14.38571429 15.04285714]]
    >>> print(y_pred)
    [ 5.28571429  7.17142857  9.05714286 10.94285714 12.82857143 14.71428571]
    """

    cv_need_agg_function_ = ["Subsample"]
    no_agg_cv_ = ["prefit", "split"]
    valid_methods_ = ["naive", "base", "plus", "minmax"]
    no_agg_methods_ = ["naive", "base"]
    valid_agg_functions_ = [None, "median", "mean"]
    ensemble_agg_functions_ = ["median", "mean"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "conformity_scores_",
        "conformity_score_function_",
        "n_features_in_",
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.conformity_score = conformity_score
        self.random_state = random_state

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        self._check_method(self.method)
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(
        self, method: str
    ) -> str:
        """
        Check if ``method`` is correct.

        Parameters
        ----------
        method: str
            Method's name to check.

        Returns
        -------
        str
            ``method`` itself.

        Raises
        ------
        ValueError
            If ``method`` is not in ``self.valid_methods_``.
        """
        if method not in self.valid_methods_:
            raise ValueError(
                f"Invalid method. Allowed values are {self.valid_methods_}."
            )
        else:
            return method

    def _check_agg_function(
        self, agg_function: Optional[str] = None
    ) -> Optional[str]:
        """
        Check if ``agg_function`` is correct, and consistent with other
        arguments.

        Parameters
        ----------
        agg_function: Optional[str]
            Aggregation function's name to check, by default ``None``.

        Returns
        -------
        str
            ``agg_function`` itself or ``"mean"``.

        Raises
        ------
        ValueError
            If ``agg_function`` is not in [``None``, ``"mean"``, ``"median"``],
            or is ``None`` while cv class is in ``cv_need_agg_function_``.
        """
        if agg_function not in self.valid_agg_functions_:
            raise ValueError(
                "Invalid aggregation function "
                f"Allowed values are '{self.valid_agg_functions_}'."
            )
        elif (agg_function is None) and (
            type(self.cv).__name__ in self.cv_need_agg_function_
        ):
            raise ValueError(
                "You need to specify an aggregation function when "
                f"cv's type is in {self.cv_need_agg_function_}."
            )
        elif (agg_function is not None) or (self.cv in self.no_agg_cv_):
            return agg_function
        else:
            return "mean"

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[RegressorMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``cv`` attribute is ``"prefit"``.
        """
        if estimator is None:
            return LinearRegression()
        else:
            check_estimator_fit_predict(estimator)
            if self.cv == "prefit":
                if isinstance(estimator, Pipeline):
                    check_is_fitted(estimator[-1])
                else:
                    check_is_fitted(estimator)
            return estimator

    def _check_ensemble(
        self, ensemble: bool,
    ) -> None:
        """
        Check if ``ensemble`` is ``False`` and if ``self.agg_function``
        is ``None``. Else raise error.

        Parameters
        ----------
        ensemble: bool
            ``ensemble`` argument to check the coherennce with
            ``self.agg_function``.

        Raises
        ------
        ValueError
            If ``ensemble`` is ``True`` and ``self.agg_function`` is ``None``.
        """
        if ensemble and (self.agg_function is None):
            raise ValueError(
                "If ensemble is True, the aggregation function has to be "
                f"in '{self.ensemble_agg_functions_}'."
            )

    def _fit_and_predict_oof_model(
        self,
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Tuple[RegressorMixin, NDArray, ArrayLike]:
        """
        Fit a single out-of-fold model on a given training set and
        perform predictions on a test set.

        Parameters
        ----------
        estimator: RegressorMixin
            Estimator to train.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        train_index: ArrayLike of shape (n_samples_train)
            Training data indices.

        val_index: ArrayLike of shape (n_samples_val)
            Validation data indices.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If ``None``, then samples are equally weighted.
            By default ``None``.

        Returns
        -------
        Tuple[RegressorMixin, NDArray, ArrayLike]

        - [0]: RegressorMixin, fitted estimator
        - [1]: NDArray of shape (n_samples_val,),
          estimator predictions on the validation fold.
        - [2]: ArrayLike of shape (n_samples_val,),
          validation data indices.
        """
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        X_val = _safe_indexing(X, val_index)
        if sample_weight is None:
            estimator = fit_estimator(estimator, X_train, y_train)
        else:
            sample_weight_train = _safe_indexing(sample_weight, train_index)
            estimator = fit_estimator(
                estimator, X_train, y_train, sample_weight_train
            )
        if _num_samples(X_val) > 0:
            y_pred = estimator.predict(X_val)
        else:
            y_pred = np.array([])
        return estimator, y_pred, val_index

    def _aggregate_with_mask(
        self,
        x: NDArray,
        k: NDArray
    ) -> NDArray:
        """
        Take the array of predictions, made by the refitted estimators,
        on the testing set, and the 1-or-nan array indicating for each training
        sample which one to integrate, and aggregate to produce phi-{t}(x_t)
        for each training sample x_t.

        Parameters:
        -----------
        x: ArrayLike of shape (n_samples_test, n_estimators)
            Array of predictions, made by the refitted estimators,
            for each sample of the testing set.

        k: ArrayLike of shape (n_samples_training, n_estimators)
            1-or-nan array: indicates whether to integrate the prediction
            of a given estimator into the aggregation, for each training
            sample.

        Returns:
        --------
        ArrayLike of shape (n_samples_test,)
            Array of aggregated predictions for each testing sample.
        """
        if self.method in self.no_agg_methods_ \
                or self.cv in self.no_agg_cv_:
            raise ValueError(
                "There should not be aggregation of predictions "
                f"if cv is in '{self.no_agg_cv_}' "
                f"or if method is in '{self.no_agg_methods_}'."
            )
        elif self.agg_function == "median":
            return phi2D(A=x, B=k, fun=lambda x: np.nanmedian(x, axis=1))
        # To aggregate with mean() the aggregation coud be done
        # with phi2D(A=x, B=k, fun=lambda x: np.nanmean(x, axis=1).
        # However, phi2D contains a np.apply_along_axis loop which
        # is much slower than the matrices multiplication that can
        # be used to compute the means.
        elif self.agg_function in ["mean", None]:
            K = np.nan_to_num(k, nan=0.0)
            return np.matmul(x, (K / (K.sum(axis=1, keepdims=True))).T)
        else:
            raise ValueError("The value of self.agg_function is not correct")

    def _pred_multi(
        self,
        X: ArrayLike
    ) -> NDArray:
        """
        Return a prediction per train sample for each test sample, by
        aggregation with matrix ``k_``.

        Parameters
        ----------
        X: NDArray of shape (n_samples_test, n_features)
            Input data

        Returns
        -------
        NDArray of shape (n_samples_test, n_samples_train)
        """
        y_pred_multi = np.column_stack(
            [e.predict(X) for e in self.estimators_]
        )
        # At this point, y_pred_multi is of shape
        # (n_samples_test, n_estimators_). The method
        # ``_aggregate_with_mask`` fits it to the right size
        # thanks to the shape of k_.
        y_pred_multi = self._aggregate_with_mask(y_pred_multi, self.k_)
        return y_pred_multi

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieRegressor:
        """
        Fit estimator and compute conformity scores used for
        prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.

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
            before the fitting process and hence have no conformity scores.
            If weights are non-uniform,
            conformity scores are still uniformly weighted.

            By default ``None``.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        estimator = self._check_estimator(self.estimator)
        agg_function = self._check_agg_function(self.agg_function)
        X, y = indexable(X, y)
        y = _check_y(y)
        sample_weight = cast(Optional[NDArray], sample_weight)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.conformity_score_function_ = check_conformity_score(
            self.conformity_score
        )
        y = cast(NDArray, y)
        n_samples = _num_samples(y)

        # Initialization
        self.estimators_: List[RegressorMixin] = []

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred = self.single_estimator_.predict(X)
            self.k_ = np.full(
                shape=(n_samples, 1), fill_value=np.nan, dtype=float
            )
        else:
            cv = cast(BaseCrossValidator, cv)
            self.k_ = np.full(
                shape=(n_samples, cv.get_n_splits(X, y)),
                fill_value=np.nan,
                dtype=float,
            )

            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight
            )

            if self.method == "naive":
                y_pred = self.single_estimator_.predict(X)
            else:
                outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_and_predict_oof_model)(
                        clone(estimator),
                        X,
                        y,
                        train_index,
                        val_index,
                        sample_weight,
                    )
                    for train_index, val_index in cv.split(X)
                )
                self.estimators_, predictions, val_indices = map(
                    list, zip(*outputs)
                )

                pred_matrix = np.full(
                    shape=(n_samples, cv.get_n_splits(X, y)),
                    fill_value=np.nan,
                    dtype=float,
                )
                for i, val_ind in enumerate(val_indices):
                    pred_matrix[val_ind, i] = np.array(
                        predictions[i], dtype=float
                    )
                    self.k_[val_ind, i] = 1
                check_nan_in_aposteriori_prediction(pred_matrix)

                y_pred = aggregate_all(agg_function, pred_matrix)

        self.conformity_scores_ = (
            self.conformity_score_function_.get_conformity_scores(y, y_pred)
        )

        if isinstance(cv, ShuffleSplit):
            self.single_estimator_ = self.estimators_[0]

        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Conformity scores from the training set and predictions
        from the model clones are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of conformity scores (``naive`` and ``base`` methods),
        - quantiles of (predictions +/- conformity scores) (``plus`` method),
        - quantiles of (max/min(predictions) +/- conformity scores)
          (``minmax`` method).

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
                - [:, 0, :]: Lower bound of the prediction interval.
                - [:, 1, :]: Upper bound of the prediction interval.
        """
        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], check_alpha(alpha))

        y_pred = self.single_estimator_.predict(X)
        n = len(self.conformity_scores_)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)
        check_alpha_and_n_samples(alpha_np, n)

        if self.method in self.no_agg_methods_ \
                or self.cv in self.no_agg_cv_:
            y_pred_multi_low = y_pred[:, np.newaxis]
            y_pred_multi_up = y_pred[:, np.newaxis]
        else:
            y_pred_multi = self._pred_multi(X)

            if self.method == "minmax":
                y_pred_multi_low = np.min(y_pred_multi, axis=1, keepdims=True)
                y_pred_multi_up = np.max(y_pred_multi, axis=1, keepdims=True)
            else:
                y_pred_multi_low = y_pred_multi
                y_pred_multi_up = y_pred_multi

            if ensemble:
                y_pred = aggregate_all(self.agg_function, y_pred_multi)

        # compute distributions of lower and upper bounds
        if self.conformity_score_function_.sym:
            conformity_scores_low = -self.conformity_scores_
            conformity_scores_up = self.conformity_scores_
        else:
            conformity_scores_low = self.conformity_scores_
            conformity_scores_up = self.conformity_scores_
            alpha_np = alpha_np / 2

        lower_bounds = (
            self.conformity_score_function_.get_estimation_distribution(
                y_pred_multi_low, conformity_scores_low
            )
        )
        upper_bounds = (
            self.conformity_score_function_.get_estimation_distribution(
                y_pred_multi_up, conformity_scores_up
            )
        )

        # get desired confidence intervals according to alpha
        y_pred_low = np.column_stack(
            [
                np_nanquantile(
                    lower_bounds.astype(float),
                    _alpha,
                    axis=1,
                    method="lower",
                )
                for _alpha in alpha_np
            ]
        )
        y_pred_up = np.column_stack(
            [
                np_nanquantile(
                    upper_bounds.astype(float),
                    1 - _alpha,
                    axis=1,
                    method="higher",
                )
                for _alpha in alpha_np
            ]
        )

        return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
