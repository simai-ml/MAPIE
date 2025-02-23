from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import (BaseCrossValidator, BaseShuffleSplit)
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.utils import check_no_agg_cv, fit_estimator, fix_number_of_classes


class EnsembleClassifier:
    """
    This class implements methods to handle the training and usage of the
    estimator. This estimator can be unique or composed by cross validated
    estimators.

    Parameters
    ----------
    estimator: Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

        By default ``None``.

    cv: Optional[str]
        The cross-validation strategy for computing scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
            If equal to -1, equivalent to
            ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
            Main variants are:
            - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
            - ``sklearn.model_selection.KFold`` (cross-validation)
        - ``"split"``, does not involve cross-validation but a division
            of the data into training and calibration subsets. The splitter
            used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
            All data provided in the ``fit`` method is then used
            to calibrate the predictions through the score computation.
            At prediction time, quantiles of these scores are used to estimate
            prediction sets.

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

    verbose: int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    single_estimator_: sklearn.ClassifierMixin
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
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "use_split_method_",
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin],
        n_classes: int,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        n_jobs: Optional[int],
        test_size: Optional[Union[int, float]],
        verbose: int,
    ):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.verbose = verbose

    @staticmethod
    def _fit_oof_estimator(
        estimator: ClassifierMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> ClassifierMixin:
        """
        Fit a single out-of-fold model on a given training set.

        Parameters
        ----------
        estimator: ClassifierMixin
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
        ClassifierMixin
            Fitted estimator.
        """
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
    def _check_proba_normalized(
        y_pred_proba: ArrayLike,
        axis: int = 1
    ) -> ArrayLike:
        """
        Check if, for all the observations, the sum of
        the probabilities is equal to one.

        Parameters
        ----------
        y_pred_proba: ArrayLike of shape
            (n_samples, n_classes) or (n_samples, n_train_samples, n_classes)
            Softmax output of a model.

        Returns
        -------
        ArrayLike of shape (n_samples, n_classes)
            Softmax output of a model if the scores all sum to one.

        Raises
        ------
        ValueError
            If the sum of the scores is not equal to one.
        """
        np.testing.assert_allclose(
            np.sum(y_pred_proba, axis=axis),
            1,
            err_msg="The sum of the scores is not equal to one.",
            rtol=1e-5
        )
        return y_pred_proba

    def _predict_proba_oof_estimator(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
        **predict_params
    ) -> NDArray:
        """
        Predict probabilities of a test set from a fitted estimator.

        Parameters
        ----------
        estimator: ClassifierMixin
            Fitted estimator.

        X: ArrayLike
            Test set.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        y_pred_proba = estimator.predict_proba(X, **predict_params)
        # we enforce y_pred_proba to contain all labels included in y
        if len(estimator.classes_) != self.n_classes:
            y_pred_proba = fix_number_of_classes(
                self.n_classes, estimator.classes_, y_pred_proba
            )
        return y_pred_proba

    def _predict_proba_calib_oof_estimator(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
        val_index: ArrayLike,
        k: int,
        **predict_params
    ) -> Tuple[NDArray, ArrayLike, ArrayLike]:
        """
        Perform predictions on a single out-of-fold model on a validation set.

        Parameters
        ----------
        estimator: ClassifierMixin
            Estimator to train.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        val_index: ArrayLike of shape (n_samples_val)
            Validation data indices.

        Returns
        -------
        Tuple[NDArray, ArrayLike]
            Predictions of estimator from val_index of X.
        """

        X_val = _safe_indexing(X, val_index)
        if _num_samples(X_val) > 0:
            y_pred_proba = self._predict_proba_oof_estimator(estimator, X_val,
                                                             **predict_params)
        else:
            y_pred_proba = np.array([])
        val_id = np.full(len(X_val), k, dtype=int)

        return y_pred_proba, val_id, val_index

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_enc: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> EnsembleClassifier:
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
        EnsembleClassifier
            The estimator fitted.
        """
        # Initialization
        single_estimator_: ClassifierMixin
        estimators_: List[ClassifierMixin] = []
        full_indexes = np.arange(_num_samples(X))
        cv = self.cv
        self.use_split_method_ = check_no_agg_cv(X, self.cv, self.no_agg_cv_)
        estimator = self.estimator
        n_samples = _num_samples(y)

        # Computation
        if cv == "prefit":
            single_estimator_ = estimator
            k_ = (
                np.full(shape=(n_samples, 1), fill_value=np.nan, dtype=float)
            )
        else:
            single_estimator_ = self._fit_oof_estimator(
                clone(estimator),
                X,
                y,
                full_indexes,
                sample_weight,
                **fit_params
            )
            cv = cast(BaseCrossValidator, cv)
            k_ = np.empty_like(y, dtype=int)

            estimators_ = Parallel(self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_oof_estimator)(
                    clone(estimator),
                    X,
                    y_enc,
                    train_index,
                    sample_weight,
                    **fit_params
                )
                for train_index, _ in cv.split(X, y, groups)
            )
            # In split-CP, we keep only the model fitted on train dataset
            if self.use_split_method_:
                single_estimator_ = estimators_[0]

        self.single_estimator_ = single_estimator_
        self.estimators_ = estimators_
        self.k_ = k_

        return self

    def predict_proba_calib(
        self,
        X: NDArray,
        y: NDArray,
        y_enc: NDArray,
        groups: Optional[NDArray] = None,
        **predict_params
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Perform predictions on X, the calibration set.

        Parameters
        ----------
        X: NDArray of shape (n_samples_test, n_features)
            Input data

        y: Optional[NDArray] of shape (n_samples_test,)
            Input labels.

            By default ``None``.

        groups: Optional[NDArray] of shape (n_samples_test,)
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
            y_pred_proba = self.single_estimator_.predict_proba(X)
            # y_pred_proba = self._check_proba_normalized(y_pred_proba)
        else:
            X = cast(NDArray, X)
            y_pred_proba = np.empty((len(X), self.n_classes), dtype=float)
            cv = cast(BaseCrossValidator, self.cv)
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._predict_proba_calib_oof_estimator)(
                    estimator, X, calib_index, k, **predict_params
                )
                for k, ((_, calib_index), estimator) in enumerate(
                    zip(cv.split(X, y, groups), self.estimators_)
                )
            )
            (predictions_list, val_ids_list, val_indices_list) = map(
                list, zip(*outputs)
            )

            predictions = np.concatenate(cast(List[NDArray], predictions_list))
            val_ids = np.concatenate(cast(List[NDArray], val_ids_list))
            val_indices = np.concatenate(cast(List[NDArray], val_indices_list))
            self.k_[val_indices] = val_ids
            y_pred_proba[val_indices] = predictions

            if isinstance(cv, BaseShuffleSplit):
                # Should delete values indices that
                # are not used during calibration
                self.k_ = self.k_[val_indices]
                y_pred_proba = y_pred_proba[val_indices]
                y_enc = y_enc[val_indices]
                y = y[val_indices]

        return y_pred_proba, y, y_enc

    def predict(
        self,
        X: ArrayLike,
        agg_scores: Optional[str] = None,
        **predict_params,
    ) -> NDArray:
        """
        Predict target from X. It also computes the prediction per train sample
        for each test sample according to ``agg_scores``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        agg_scores: Optional[str]
            How to aggregate the scores output by the estimators on test data
            if a cross-validation strategy is used

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        NDArray
            Predictions of shape
            (n_samples, n_classes)
        """
        check_is_fitted(self, self.fit_attributes)

        if self.cv == "prefit":
            y_pred_proba = self.single_estimator_.predict_proba(
                X, **predict_params
            )
        else:
            y_pred_proba_k = np.asarray(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._predict_proba_oof_estimator)(
                        estimator, X, **predict_params
                    ) for estimator in self.estimators_
                )
            )
            if agg_scores == "crossval":
                y_pred_proba = np.moveaxis(y_pred_proba_k[self.k_], 0, 2)
            elif agg_scores == "mean":
                y_pred_proba = np.mean(y_pred_proba_k, axis=0)
            else:
                raise ValueError("Invalid 'agg_scores' argument.")

        return y_pred_proba
