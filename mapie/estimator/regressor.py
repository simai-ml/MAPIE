from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import _safe_indexing, deprecated
from sklearn.utils.validation import _num_samples, check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all, phi2D
from mapie.utils import (check_nan_in_aposteriori_prediction, check_no_agg_cv,
                         fit_estimator)


class EnsembleRegressor:
    """
    This class implements methods to handle the training and usage of the
    estimator. This estimator can be unique or composed by cross validated
    estimators.

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

    Attributes
    ----------
    single_estimator_: sklearn.RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: list
        List of out-of-folds estimators.

    k_: ArrayLike
        - Array of nans, of shape (len(y), 1) if ``cv`` is ``"prefit"``
            (defined but not used)
        - Dummy array of folds containing each training sample, otherwise.
            Of shape (n_samples_train, cv.get_n_splits(X_train, y_train)).
    """
    no_agg_cv_ = ["prefit", "split"]
    no_agg_methods_ = ["naive", "base"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "use_split_method_",
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin],
        method: str,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        agg_function: Optional[str],
        n_jobs: Optional[int],
        test_size: Optional[Union[int, float]],
        verbose: int
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.agg_function = agg_function
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.verbose = verbose

    @staticmethod
    def _fit_oof_estimator(
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> RegressorMixin:
        """
        Fit a single out-of-fold model on a given training set.

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

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        RegressorMixin
            Fitted estimator.
        """
        # TODO back-end: avoid using private utilities from sklearn like
        # _safe_indexing (may break anytime without notice)
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        if not (sample_weight is None):
            sample_weight = _safe_indexing(sample_weight, train_index)
            sample_weight = cast(NDArray, sample_weight)

        estimator = fit_estimator(
            estimator,
            X_train,
            y_train,
            sample_weight=sample_weight,
            **fit_params
        )
        return estimator

    @staticmethod
    def _predict_oof_estimator(
        estimator: RegressorMixin,
        X: ArrayLike,
        val_index: ArrayLike,
        **predict_params
    ) -> Tuple[NDArray, ArrayLike]:
        """
        Perform predictions on a single out-of-fold model on a validation set.

        Parameters
        ----------
        estimator: RegressorMixin
            Estimator to train.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        val_index: ArrayLike of shape (n_samples_val)
            Validation data indices.

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Tuple[NDArray, ArrayLike]
            Predictions of estimator from val_index of X.
        """
        X_val = _safe_indexing(X, val_index)
        if _num_samples(X_val) > 0:
            y_pred = estimator.predict(X_val, **predict_params)
        else:
            y_pred = np.array([])
        return y_pred, val_index

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

        Parameters
        ----------
        x: ArrayLike of shape (n_samples_test, n_estimators)
            Array of predictions, made by the refitted estimators,
            for each sample of the testing set.

        k: ArrayLike of shape (n_samples_training, n_estimators)
            1-or-nan array: indicates whether to integrate the prediction
            of a given estimator into the aggregation, for each training
            sample.

        Returns
        -------
        ArrayLike of shape (n_samples_test,)
            Array of aggregated predictions for each testing sample.
        """
        if self.method in self.no_agg_methods_ or self.use_split_method_:
            raise ValueError(
                "There should not be aggregation of predictions."
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
            raise ValueError(
                "The value of the aggregation function is not correct"
            )

    def _pred_multi(self, X: ArrayLike, **predict_params) -> NDArray:
        """
        Return a prediction per train sample for each test sample, by
        aggregation with matrix ``k_``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        NDArray of shape (n_samples_test, n_samples_train)
        """
        y_pred_multi = np.column_stack(
            [e.predict(X, **predict_params) for e in self.estimators_]
        )
        # At this point, y_pred_multi is of shape
        # (n_samples_test, n_estimators_). The method
        # ``_aggregate_with_mask`` fits it to the right size
        # thanks to the shape of k_.
        y_pred_multi = self._aggregate_with_mask(y_pred_multi, self.k_)
        return y_pred_multi

    def predict_calib(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **predict_params
    ) -> NDArray:
        """
        Perform predictions on X : the calibration set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data

        y: Optional[ArrayLike] of shape (n_samples_test,)
            Input labels.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples_test,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        NDArray of shape (n_samples_test, 1)
            The predictions.
        """
        check_is_fitted(self, self.fit_attributes)

        if self.cv == "prefit":
            y_pred = self.single_estimator_.predict(X)
        else:
            if self.method == "naive":
                y_pred = self.single_estimator_.predict(X)
            else:
                cv = cast(BaseCrossValidator, self.cv)
                outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._predict_oof_estimator)(
                        estimator, X, calib_index, **predict_params
                    )
                    for (_, calib_index), estimator in zip(
                        cv.split(X, y, groups),
                        self.estimators_
                    )
                )
                predictions, indices = map(
                    list, zip(*outputs)
                )
                n_samples = _num_samples(X)
                pred_matrix = np.full(
                    shape=(n_samples, cv.get_n_splits(X, y, groups)),
                    fill_value=np.nan,
                    dtype=float,
                )
                for i, ind in enumerate(indices):
                    pred_matrix[ind, i] = np.array(
                        predictions[i], dtype=float
                    )
                    self.k_[ind, i] = 1

                if self.use_split_method_:
                    y_pred = pred_matrix.flatten()
                else:
                    check_nan_in_aposteriori_prediction(pred_matrix)
                    y_pred = aggregate_all(self.agg_function, pred_matrix)

        return y_pred

    @deprecated(
        "WARNING: EnsembleRegressor.fit is deprecated."
        "Instead use EnsembleRegressor.fit_single_estimator"
        "then EnsembleRegressor.fit_multi_estimators"
    )
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params
    ) -> EnsembleRegressor:
        """
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        EnsembleRegressor
            The estimator fitted.
        """
        self.fit_single_estimator(
            X,
            y,
            sample_weight,
            groups,
            **fit_params
        )

        self.fit_multi_estimators(
            X,
            y,
            sample_weight,
            groups,
            **fit_params
        )

        return self

    def fit_multi_estimators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params
    ) -> EnsembleRegressor:

        n_samples = _num_samples(y)
        estimators: List[RegressorMixin] = []

        if self.cv == "prefit":

            # Create a placeholder attribute 'k_' filled with NaN values
            # This attribute is defined for consistency but
            # is not used in prefit mode
            self.k_ = np.full(
                shape=(n_samples, 1), fill_value=np.nan, dtype=float
            )

        else:
            cv = cast(BaseCrossValidator, self.cv)
            self.k_ = np.full(
                shape=(n_samples, cv.get_n_splits(X, y, groups)),
                fill_value=np.nan,
                dtype=float,
            )

            if self.method != "naive":
                estimators = Parallel(
                    self.n_jobs,
                    verbose=self.verbose
                )(
                    delayed(self._fit_oof_estimator)(
                        clone(self.estimator),
                        X,
                        y,
                        train_index,
                        sample_weight,
                        **fit_params
                    )
                    for train_index, _ in cv.split(X, y, groups)
                )

        self.estimators_ = estimators

        return self

    def fit_single_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params
    ) -> EnsembleRegressor:

        self.use_split_method_ = check_no_agg_cv(X, self.cv, self.no_agg_cv_)
        single_estimator_: RegressorMixin

        if self.cv == "prefit":
            single_estimator_ = self.estimator
        else:
            cv = cast(BaseCrossValidator, self.cv)
            if self.use_split_method_:
                train_indexes = [
                    train_index for train_index, test_index in cv.split(
                        X, y, groups)
                ][0]
                indexes = train_indexes
            else:
                full_indexes = np.arange(_num_samples(X))
                indexes = full_indexes

            single_estimator_ = self._fit_oof_estimator(
                    clone(self.estimator),
                    X,
                    y,
                    indexes,
                    sample_weight,
                    **fit_params
                )

        self.single_estimator_ = single_estimator_
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        return_multi_pred: bool = True,
        **predict_params
    ) -> Union[NDArray, Tuple[NDArray, NDArray, NDArray]]:
        """
        Predict target from X. It also computes the prediction per train sample
        for each test sample according to ``self.method``.

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

        return_multi_pred: bool
            If ``True`` the method returns the predictions and the multiple
            predictions (3 arrays). If ``False`` the method return the
            simple predictions only.

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            - Predictions
            - The multiple predictions for the lower bound of the intervals.
            - The multiple predictions for the upper bound of the intervals.
        """
        check_is_fitted(self, self.fit_attributes)

        y_pred = self.single_estimator_.predict(X, **predict_params)
        if not return_multi_pred and not ensemble:
            return y_pred

        if self.method in self.no_agg_methods_ or self.use_split_method_:
            y_pred_multi_low = y_pred[:, np.newaxis]
            y_pred_multi_up = y_pred[:, np.newaxis]
        else:
            y_pred_multi = self._pred_multi(X, **predict_params)

            if self.method == "minmax":
                y_pred_multi_low = np.min(y_pred_multi, axis=1, keepdims=True)
                y_pred_multi_up = np.max(y_pred_multi, axis=1, keepdims=True)
            elif self.method == "plus":
                y_pred_multi_low = y_pred_multi
                y_pred_multi_up = y_pred_multi
            elif self.method == "enbpi":
                y_pred_aggregate = aggregate_all(
                    self.agg_function, y_pred_multi)
                y_pred_multi_low = y_pred_aggregate[:, np.newaxis]
                y_pred_multi_up = y_pred_aggregate[:, np.newaxis]
            else:
                y_pred_multi_low = y_pred[:, np.newaxis]
                y_pred_multi_up = y_pred[:, np.newaxis]

            if ensemble:
                y_pred = aggregate_all(self.agg_function, y_pred_multi)

        if return_multi_pred:
            return y_pred, y_pred_multi_low, y_pred_multi_up
        else:
            return y_pred
