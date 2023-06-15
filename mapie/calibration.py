from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from ._typing import ArrayLike, NDArray
from .utils import (check_estimator_classification,
                    check_estimator_fit_predict, check_n_features_in,
                    check_null_weight, fit_estimator, get_calib_set)


class MapieCalibrator(BaseEstimator, ClassifierMixin):
    """
    Calibration for multi-class problems.

    This class performs calibration for various methods, currently only
    top-label calibration [1].

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: Optional[str]
        Method to choose for calibration method.
        Choose among:

        - "top_label", performs a calibration on the class with highest score
        given both score and class, see section 2 of [1].

        By default "top_label".

    calibrator : Optional[Union[str, RegressorMixin]]
        Any calibrator with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, calibrator defaults to a string "sigmoid" instance.

        By default ``None``.

    cv: Optional[str]
        The cross-validation strategy to compute scores :

        - "split", performs a standard splitting into a calibration and a
        test set.
        - "prefit", assumes that ``estimator`` has been fitted already.
          All the data that are provided in the ``fit`` method are then used
          to calibrate the predictions through the score computation.

        By default "split".

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    classes_: NDArray
        Array with the name of each class.

    n_classes_: int
        Number of classes that are in the training dataset.

    uncalib_pred: NDArray
        Array of the uncalibrated predictions set by the ``estimator``.

    single_estimator_: ClassifierMixin
        Classifier fitted on the training data.

    calibrators: Dict[Union[int, str], RegressorMixin]
        Dictionnary of all the fitted calibrators.

    References
    ----------
    [1] Gupta, Chirag, and Aaditya K. Ramdas. "Top-label calibration
    and multiclass-to-binary reductions." arXiv preprint
    arXiv:2107.08353 (2021).


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.calibration import MapieCalibrator
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> mapie = MapieCalibrator().fit(X_toy, y_toy, random_state=20)
    >>> y_calib = mapie.predict_proba(X_toy)
    >>> print(y_calib)
    [[0.84900723        nan        nan]
     [0.75432411        nan        nan]
     [0.62285341        nan        nan]
     [       nan 0.33333333        nan]
     [       nan 0.33333333        nan]
     [       nan 0.33333333        nan]
     [       nan        nan 0.33333002]
     [       nan        nan 0.54326683]
     [       nan        nan 0.66666124]]
    """

    fit_attributes = [
        "estimator",
        "calibrators",
    ]

    named_calibrators = {
        "sigmoid": _SigmoidCalibration(),
        "isotonic": IsotonicRegression(out_of_bounds="clip")
    }

    valid_methods = ["top_label"]

    valid_cv = ["prefit", "split"]

    valid_inputs = ["multiclass", "binary"]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "top_label",
        calibrator: Optional[Union[str, RegressorMixin]] = None,
        cv: Optional[str] = "split",
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.calibrator = calibrator
        self.cv = cv

    def _check_cv(
        self,
        cv: Optional[str],
    ) -> str:
        """
        Check if cross-validator is ``"prefit"`` or ``"split"``.
        Else raise error.

        Parameters
        ----------
        cv : str
            Cross-validator to check.

        Returns
        -------
        str
            'prefit' or 'split'.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv in self.valid_cv:
            return cv
        raise ValueError(
            "Invalid cv argument. "
            f"Allowed values are {self.valid_cv}."
        )

    def _check_calibrator(
        self,
        calibrator: Optional[Union[str, RegressorMixin]],
    ) -> RegressorMixin:
        """
        Check the input that has been provided for calibrator and
        check that the calibrator is a valid estimator to calibrate.

        Parameters
        ----------
        calibrator : Union[str, RegressorMixin]
            If calibrator is a string then it returns the corresponding
            estimator of ``named_calibrators``, else returns calibrator.

            By defaults ``None``. If ``None``, defaults to ``"sigmoid"``.

        Returns
        -------
        RegressorMixin
            RegressorMixin calibrator.

        Raises
        ------
        ValueError
            If calibrator is not a key of ``named_calibrators``.
        """
        if calibrator is None:
            calibrator = "sigmoid"

        if isinstance(calibrator, str):
            if calibrator in self.named_calibrators.keys():
                calibrator = self.named_calibrators[calibrator]
            else:
                raise ValueError(
                    "Please provide a string in: "
                    + (", ").join(self.named_calibrators.keys()) + "."
                )
        check_estimator_fit_predict(calibrator)
        return calibrator

    def _get_labels(
        self,
        X: ArrayLike
    ) -> Tuple[NDArray, NDArray]:
        """
        This method depends on the value of ``method`` and collects the labels
        that are needed to transform a multi-class calibration to multiple
        binary calibrations.

        - "top_label" method means that you take the maximum score and
        calibrate each maximum class separately in a one-versus all setting,
        see section 2 of [1].

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        Tuple[NDArray, NDArray] of shapes (n_samples,) and (n_samples,)
            The first element corresponds to the output scores that have
            to be calibrated and the second element corresponds to the
            predictions.

            In the "top_label" setting, this refers to the maximum scores
            and the class associated to the maximum score.
        """
        pred = self.single_estimator_.predict_proba(X=X)
        max_class_prob = np.max(pred, axis=1).reshape(-1, 1)
        y_pred = self.classes_[np.argmax(pred, axis=1)]
        return max_class_prob, y_pred

    def _check_method(self) -> None:
        """
        Check that the method is valid.

        Raises
        ------
        ValueError
            If the method does not belong to the valid methods.
        """
        if self.method not in self.valid_methods:
            raise ValueError(
                "Invalid method, allowed method are: "
                + (", ").join(self.valid_methods) + "."
            )

    def _check_type_of_target(self, y: ArrayLike):
        """
        Check type of target for calibration class.

        Parameters
        ----------
        y : ArrayLike of shape (n_samples,)
            Training labels.
        """
        if type_of_target(y) not in self.valid_inputs:
            raise ValueError(
                "Make sure to have one of the allowed targets: "
                + (", ").join(self.valid_inputs) + "."
            )

    def _fit_calibrator(
        self,
        label: Union[int, str],
        calibrator: RegressorMixin,
        y_calib: NDArray,
        top_class_prob: NDArray,
        y_pred: NDArray,
        sample_weight: Optional[ArrayLike],
    ) -> RegressorMixin:
        """
        Fitting the calibrator requires that we have a binary target, hence,
        we find the subset of values on which we want to apply this
        calibration and apply a binary calibration on these.

        Parameters
        ----------
        label : Union[int, str]
            The class for which we fit a calibrator.
        calibrator : RegressorMixin
            Calibrator to fit.
        y_calib : NDArray of shape (n_samples,)
            Training labels.
        top_class_prob : NDArray of shape (n_samples,)
            The highest score for each input of the method ``predict_proba``
            of the ``estimator`.
        y_pred : NDArray of shape (n_samples,)
            Predictions.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        RegressorMixin
            Calibrated estimator.
        """
        calibrator_ = clone(calibrator)
        sample_weight = cast(NDArray, sample_weight)
        given_label_indices = np.argwhere(y_pred.ravel() == label).ravel()
        y_calib_ = np.equal(y_calib[given_label_indices], label).astype(int)
        top_class_prob_ = top_class_prob[given_label_indices]

        if sample_weight is not None:
            sample_weight_ = sample_weight[given_label_indices]
            (
                sample_weight_, top_class_prob_, y_calib_
            ) = check_null_weight(
                sample_weight_,
                top_class_prob_,
                y_calib_
            )
        else:
            sample_weight_ = sample_weight
        calibrator_ = fit_estimator(
            calibrator_, top_class_prob_, y_calib_, sample_weight_
        )
        return calibrator_

    def _fit_calibrators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike],
        calibrator: RegressorMixin,
    ) -> Dict[Union[int, str], RegressorMixin]:
        """
        This method sequentially fits the calibrators for each labels
        defined by ``_get_labels`` method.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration.
            By default ``None``.
        calibrator : RegressorMixin
            Calibrator to fit.

        Returns
        -------
        Dict[Union[int, str], RegressorMixin]
            Dictionnary of fitted calibrators.
        """
        X, y = indexable(X, y)
        y = _check_y(y)
        max_prob, y_pred = self._get_labels(X)
        calibrators = {}
        for label in np.unique(y_pred):
            calibrator_ = self._fit_calibrator(
                label,
                calibrator,
                cast(NDArray, y),
                max_prob,
                y_pred,
                sample_weight,
            )
            calibrators[label] = calibrator_
        return calibrators

    def _pred_proba_calib(
        self,
        idx: int,
        label: Union[int, str],
        calibrated_values: NDArray,
        max_prob: NDArray,
        y_pred: NDArray,
    ) -> None:
        """
        Using the predicted scores, we calibrate the maximum score with the
        specifically fitted calibrator. Note that if there is no calibrator
        for the specific class, then we simply output the not calibrated
        values.


        Parameters
        ----------
        idx : int
            Position of the label for an enumerate of all labels.
        label : Union[int, str]
            The label to define the subset of values to be taken into account.
        calibrated_values : NDArray
            Array of calibrated values to be updated.
        max_prob : NDArray of shape (n_samples,)
            Values to be calibrated.
        y_pred : NDArray of shape (n_samples,)
            Indicator of the values to be calibrated.


        Raises
        ------
        Warnings
            If there has not been a fitted calibrator for the specific class.
        """
        idx_labels = np.where(y_pred.ravel() == label)[0].ravel()
        if label not in self.calibrators.keys():
            calibrated_values[
                idx_labels, idx
                ] = max_prob[idx_labels].ravel()
            warnings.warn(
                f"WARNING: This predicted label {label} has not been seen "
                + " during the calibration and therefore scores will remain"
                + " unchanged."
            )
        else:
            calibrator_ = self.calibrators[label]
            preds_ = calibrator_.predict(max_prob[idx_labels])
            calibrated_values[idx_labels, idx] = preds_

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        calib_size: Optional[float] = 0.33,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> MapieCalibrator:
        """
        Calibrate the estimator on given datasets, according to the chosen
        method.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        calib_size : Optional[float]
            If ``cv == split`` and X_calib and y_calib are not defined, then
            the calibration dataset is created with the split defined by
            calib_size.
        random_state : int, RandomState instance or ``None``, default is
            ``None``
            See ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
        shuffle : bool, default=True
            See ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If shuffle=False, then stratify must be ``None``.
        stratify : array-like, default=None
            See ``sklearn.model_selection.train_test_split`` documentation.
            If not ``None``, data is split in a stratified fashion, using this
            as the class label.

        Returns
        -------
        MapieCalibrator
            The model itself.
        """
        self._check_method()
        cv = self._check_cv(self.cv)
        X, y = indexable(X, y)
        y = _check_y(y)
        self._check_type_of_target(y)
        estimator = check_estimator_classification(X, y, cv, self.estimator)
        calibrator = self._check_calibrator(self.calibrator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        if cv == "prefit":
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)
            self.calibrators = self._fit_calibrators(
                X, y, sample_weight, calibrator
            )
        if cv == "split":
            results = get_calib_set(
                X,
                y,
                sample_weight=sample_weight,
                calib_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
            X_train, y_train, X_calib, y_calib, sw_train, sw_calib = results
            X_train, y_train = indexable(X_train, y_train)
            y_train = _check_y(y_train)
            sw_train, X_train, y_train = check_null_weight(
                sw_train,
                X_train,
                y_train
            )
            estimator = fit_estimator(
                clone(estimator), X_train, y_train, sw_train,
            )
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)
            self.calibrators = self._fit_calibrators(
                X_calib, y_calib, sw_calib, calibrator
            )
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction of the calibrated scores using fitted classifier and
        calibrator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            The calibrated score for each max score and zeros at every
            other position in that line.
        """
        check_is_fitted(self, self.fit_attributes)
        self.uncalib_pred = self.single_estimator_.predict_proba(X=X)

        max_prob, y_pred = self._get_labels(X)

        n = _num_samples(max_prob)
        calibrated_test_values = np.full((n, self.n_classes_), np.nan)

        for idx, label in enumerate(np.unique(y_pred)):
            self._pred_proba_calib(
                idx,
                label,
                calibrated_test_values,
                max_prob,
                y_pred,
            )
        return calibrated_test_values

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Predict the class of the estimator after calibration.
        Note that in the top-label setting, this class does not change.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples,)
            The class from the scores.
        """
        check_is_fitted(self, self.fit_attributes)
        return self.single_estimator_.predict(X)
