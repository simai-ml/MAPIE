from __future__ import annotations

import warnings
from itertools import chain
from typing import Iterable, Optional, Sequence, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from ._typing import ArrayLike, NDArray
from .control_risk.crc_rcps import find_lambda_star, get_r_hat_plus
from .control_risk.ltt import find_lambda_control_star, ltt_procedure
from .control_risk.risks import compute_risk_precision, compute_risk_recall
from .utils import check_alpha, check_n_jobs, check_verbose


class MapieMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction methods for
    estimating prediction sets for multilabel-classification.
    It guarantees (under the hypothesis of exchangeability) that
    a risk is at least 1 - alpha (alpha is a user-specified parameter).
    For now, we consider the recall as risk.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any fitted multi-label classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods).
        If ``None``, estimator by default is a sklearn LogisticRegression
        instance.

         by default ``None``

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        For this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        "None" is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        to evaluate quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.

        By default ``1``.

    verbose : int, optional
        The verbosity level, used with joblib for parallel processing.
        For the moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods. Either CRC or RCPS
    valid_methods: List[Union[str, ``None``]]
        List of all valid bounds computation for RCPS only.
    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_lambdas: int
        Number of thresholds on which we compute the risk.

    lambdas: NDArray
        Array with all the values of lambda.

    risks : ArrayLike of shape (n_samples_cal, n_lambdas)
        The risk for each observation for each threshold

    r_hat : ArrayLike of shape (n_lambdas)
        Average risk for each lambda

    r_hat_plus: ArrayLike of shape (n_lambdas)
        Upper confidence bound for each lambda, computed
        with different bounds (see predict). Only relevant when
        method="rcps".

    lambdas_star: ArrayLike of shape (n_lambdas)
        Optimal threshold for a given alpha.

    valid_index: List[List[Any]]
        List of list of all index that satisfy fwer controlling.
        This attribute is computed when the user wants to
        control precision score.
        Only relevant when metric_control="precision" as it uses
        learn then test (ltt) procedure.
        Contains n_alpha lists (see predict).

     sigma_init : Optional[float]
        First variance in the sigma_hat array. The default
        value is the same as in the paper implementation [1].

    References
    ----------
    [1] Lihua Lei Jitendra Malik Stephen Bates, Anastasios Angelopoulos
    and Michael I. Jordan. Distribution-free, risk-controlling prediction
    sets. CoRR, abs/2101.02703, 2021.
    URL https://arxiv.org/abs/2101.02703.39

    [2] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua,
    Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

    [3] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mapie.multi_label_classification import MapieMultiLabelClassifier
    >>> X_toy = np.arange(4).reshape(-1, 1)
    >>> y_toy = np.stack([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    >>> mapie = MapieMultiLabelClassifier(estimator=clf).fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.3)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False  True]
     [ True False  True]
     [False  True False]
     [False  True False]]
    """
    valid_methods_by_metric_ = {
        "precision": ["ltt"],
        "recall": ["rcps", "crc"]
    }
    valid_methods = list(chain(*valid_methods_by_metric_.values()))
    valid_metric_ = list(valid_methods_by_metric_.keys())
    valid_bounds_ = ["hoeffding", "bernstein", "wsr", None]
    lambdas = np.arange(0, 1, 0.01)
    n_lambdas = len(lambdas)
    fit_attributes = [
        "single_estimator_",
        "risks"
    ]
    sigma_init = 0.25  # Value given in the paper [1]
    cal_size = .3

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        metric_control: Optional[str] = 'recall',
        method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.metric_control = metric_control
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Check n_jobs, verbose and random_states.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(self) -> None:
        """
        Check that the specified method is valid

        Raises
        ------
        ValueError
            Raise error if the name of the method is not
            in self.valid_methods_
        """
        self.method = cast(str, self.method)
        self.metric_control = cast(str, self.metric_control)

        if self.method not in self.valid_methods_by_metric_[
            self.metric_control
        ]:
            raise ValueError(
                "Invalid method for metric: "
                + "You are controlling " + self.metric_control
                + " and you are using invalid method: " + self.method
                + ". Use instead: " + "".join(self.valid_methods_by_metric_[
                    self.metric_control]
                )
            )

    def _check_all_labelled(self, y: NDArray) -> None:
        """
        Check that all observations have at least
        one label

        Parameters
        ----------
        y : NDArray of shape (n_samples, n_labels)
            Labels of the observations.

        Raises
        ------
        ValueError
            Raise error if at least one observation
            has no label.
        """
        if not (y.sum(axis=1) > 0).all():
            raise ValueError(
                "Invalid y. "
                "All observations should contain at "
                "least one label."
            )

    def _check_delta(self, delta: Optional[float]):
        """
        Check that delta is not ``None`` when the
        method is RCPS and that it is between 0 and 1.

        Parameters
        ----------
        delta : float
            Probability with which we control the risk. The higher
            the probability, the more conservative the algorithm will be.

        Raises
        ------
        ValueError
            If delta is ``None`` and method is RCSP or
            if delta is not in [0, 1] and method
            is RCPS.
        Warning
            If delta is not ``None`` and method is CRC
        """
        if (not isinstance(delta, float)) and (delta is not None):
            raise ValueError(
                "Invalid delta. "
                f"delta must be a float, not a {type(delta)}"
            )
        if (self.method == "rcps") or (self.method == "ltt"):
            if delta is None:
                raise ValueError(
                    "Invalid delta. "
                    "delta cannot be ``None`` when controlling "
                    "Recall with RCPS or Precision with LTT"
                )
            elif ((delta <= 0) or (delta >= 1)):
                raise ValueError(
                    "Invalid delta. "
                    "delta must be in ]0, 1["
                )
        if (self.method == "crc") and (delta is not None):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the delta is not ``None``, it won't be"
                + "taken into account"
            )

    def _check_valid_index(self, alpha: NDArray):
        """
        Check if the valid index is empty.
        If it is, we should warn the user that for the alpha value
        and delta level chosen, LTT will return no results.
        The user must be less inclined to take risks or
        must choose a higher alpha value.
        """
        for i in range(len(self.valid_index)):
            if self.valid_index[i] == []:
                warnings.warn(
                    "Warning: LTT method has returned an empty sequence"
                    + " for confidence level=" + str(1-alpha[i])
                )

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
        _refit: Optional[bool] = False,
    ) -> ClassifierMixin:
        """
        Check the estimator value. If it is ``None``,
        it returns a multi-output ``LogisticRegression``
        instance if necessary.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        _refit : Optional[bool]
            Whether or not the user is using fit (True) or
            partial_fit (False).

            By default False


        Returns
        -------
        ClassifierMixin
            The estimator itself or a default multi-output
            ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted.

        Warning
            If estimator is then to warn about the split of the
            data between train and calibration
        """
        if (estimator is None) and (not _refit):
            raise ValueError(
                "Invalid estimator with partial_fit. "
                "If the estimator is ``None`` you can not "
                "use partial_fit."
            )
        if (estimator is None) and (_refit):
            estimator = MultiOutputClassifier(
                LogisticRegression(multi_class="multinomial")
            )
            X_train, X_calib, y_train, y_calib = train_test_split(
                    X,
                    y,
                    test_size=self.calib_size,
                    random_state=self.random_state,
            )
            estimator.fit(X_train, y_train)
            warnings.warn(
                "WARNING: To avoid overffiting, X has been splitted"
                + "into X_train and X_conf. The conformalization will only"
                + "be done on X_conf"
            )
            return estimator, X_calib, y_calib

        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        if (
            not hasattr(est, "fit")
            or not hasattr(est, "predict")
            or not hasattr(est, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        check_is_fitted(est)
        return estimator, X, y

    def _check_partial_fit_first_call(self) -> bool:
        """
        Check that this is the first time partial_fit
        or fit is called.

        Returns
        -------
        bool
            True if it is the first time, else False.
        """
        return not hasattr(self, "risks")

    def _check_bound(self, bound: Optional[str]):
        """
        Check the value of the bound.

        Parameters
        ----------
        bound : Optional[str]
            Bound defined in the predict.

        Raises
        ------
        AttributeError
            If bound is not in ["hoeffding", "bernstein", "wsr", ``None``]

        Warning
            If bound is not ``None``and method is CRC
        """
        if bound not in self.valid_bounds_:
            raise ValueError(
                "bound must be in ['hoeffding', 'bernstein', 'wsr', ``None``]"
            )
        elif (bound is not None) and (self.method == "crc"):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the bound is not ``None``, it won't be"
                + "taken into account."
            )
        elif (bound is not None) and (self.method == "ltt"):
            warnings.warn(
                "WARNING: you are using ltt method hence "
                + "even if bound is not ``None``, it won't be"
                + "taken into account."
            )

    def _check_metric_control(self):
        """
        Check that the metrics to control are valid
        (can be a string or list of string.)
        """
        if self.metric_control not in self.valid_metric_:
            raise ValueError(
                "Invalid metric. "
                "Allowed scores must be in the following list "
                + ", ".join(self.valid_metric_)
            )

        if self.method is None:
            if self.metric_control == "recall":
                self.method = "crc"
            else:  # self.metric_control == "precision"
                self.method = "ltt"

    def _transform_pred_proba(
        self,
        y_pred_proba: Union[Sequence[NDArray], NDArray]
    ) -> NDArray:
        """If the output of the predict_proba is a list of arrays (output of
        the ``predict_proba`` of ``MultiOutputClassifier``) we transform it
        into an array of shape (n_samples, n_classes, 1), otherwise, we add
        one dimension at the end.

        Parameters
        ----------
        y_pred_proba : Union[List, NDArray]
            Output of the multi-label classifier.

        Returns
        -------
        NDArray of shape (n_samples, n_classe, 1)
            Output of the model ready for risk computation.
        """
        if isinstance(y_pred_proba, np.ndarray):
            y_pred_proba_array = y_pred_proba
        else:
            y_pred_proba_stacked = np.stack(
                y_pred_proba,  # type: ignore
                axis=0
            )[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        _refit: Optional[bool] = False,
    ) -> MapieMultiLabelClassifier:
        """
        Fit the base estimator or use the fitted base estimator on
        batch data. All the computed risks will be concatenated each
        time the partial_fit method is called.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : NDArray of shape (n_samples, n_classes)
            Training labels.

        _refit: bool
            Whether or not refit from scratch.

            By default False

        Returns
        -------
        MapieMultiLabelClassifier
            The model itself.
        """
        # Checks
        first_call = self._check_partial_fit_first_call()
        self._check_parameters()
        self._check_metric_control()
        self._check_method()

        X, y = indexable(X, y)
        _check_y(y, multi_output=True)
        estimator, X, y = self._check_estimator(
            X, y, self.estimator,
            _refit
        )

        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self._check_all_labelled(y)
        self.n_samples_ = _num_samples(X)

        # Work
        if first_call or _refit:
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            self.theta_ = X.shape[1]

            if self.metric_control == "recall":
                self.risks = compute_risk_recall(
                    self.lambdas, y_pred_proba_array, y
                )
            else:  # self.metric_control == "precision"
                self.risks = compute_risk_precision(
                    self.lambdas, y_pred_proba_array, y
                )
        else:
            if X.shape[1] != self.theta_:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_))
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            if self.metric_control == "recall":
                partial_risk = compute_risk_recall(
                    self.lambdas,
                    y_pred_proba_array,
                    y
                )
            else:  # self.metric_control == "precision"
                partial_risk = compute_risk_precision(
                    self.lambdas,
                    y_pred_proba_array,
                    y
                )
            self.risks = np.concatenate([self.risks, partial_risk], axis=0)

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        calib_size: Optional[float] = .3
    ) -> MapieMultiLabelClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: NDArray of shape (n_samples, n_classes)
            Training labels.

        calib_size: Optional[float]
            Size of the calibration dataset with respect to X if the
            given model is ``None`` need to fit a LogisticRegression.

            By default .3

        Returns
        -------
        MapieMultiLabelClassifier
            The model itself.
        """
        self.calib_size = calib_size
        return self.partial_fit(X, y, _refit=True)

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[float] = None,
        bound: Optional[Union[str, None]] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)

        alpha : Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            sets.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        delta : Optional[float]
            Can be a float, or ``None``. If using method="rcps", then it
            can not be set to ``None``.
            Between 0 and 1, the level of certainty at which we compute
            the Upper Confidence Bound of the average risk.
            Lower ``delta`` produce larger (more conservative) prediction
            sets.
            By default ``None``.

        bound : Optional[Union[str, ``None``]]
            Method used to compute the Upper Confidence Bound of the
            average risk. Only necessary with the RCPS method.
            By default ``None``.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is ``None``.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples, n_classes) and (n_samples, n_classes, n_alpha)
        if alpha is not ``None``.
        """

        self._check_delta(delta)
        self._check_bound(bound)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        check_is_fitted(self, self.fit_attributes)

        # Estimate prediction sets
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)

        y_pred_proba = self.single_estimator_.predict_proba(X)

        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
        y_pred_proba_array = np.repeat(
            y_pred_proba_array,
            len(alpha_np),
            axis=2
        )
        if self.metric_control == 'precision':
            self.n_obs = len(self.risks)
            self.r_hat = self.risks.mean(axis=0)
            self.valid_index, self.p_values = ltt_procedure(
                self.r_hat, alpha_np, delta, self.n_obs
            )
            self._check_valid_index(alpha_np)
            self.lambdas_star, self.r_star = find_lambda_control_star(
               self.r_hat, self.valid_index, self.lambdas
            )
            y_pred_proba_array = (
                y_pred_proba_array >
                np.array(self.lambdas_star)[np.newaxis, np.newaxis, :]
            )

        else:
            self.r_hat, self.r_hat_plus = get_r_hat_plus(
                self.risks, self.lambdas, self.method,
                bound, delta, self.sigma_init
            )
            self.lambdas_star = find_lambda_star(
                self.lambdas, self.r_hat_plus, alpha_np
            )
            y_pred_proba_array = (
                y_pred_proba_array >
                self.lambdas_star[np.newaxis, np.newaxis, :]
            )
        return y_pred, y_pred_proba_array
