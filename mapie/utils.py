import warnings
from inspect import signature
from typing import Any, Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _check_sample_weight, _num_features

from ._compatibility import np_quantile
from ._typing import ArrayLike, NDArray
from .conformity_scores import AbsoluteConformityScore, ConformityScore


def check_null_weight(
    sample_weight: Optional[ArrayLike], X: ArrayLike, y: ArrayLike
) -> Tuple[Optional[NDArray], ArrayLike, ArrayLike]:
    """
    Check sample weights and remove samples with null sample weights.

    Parameters
    ----------
    sample_weight : Optional[ArrayLike] of shape (n_samples,)
        Sample weights.
    X : ArrayLike of shape (n_samples, n_features)
        Training samples.
    y : ArrayLike of shape (n_samples,)
        Training labels.

    Returns
    -------
    sample_weight : Optional[NDArray] of shape (n_samples,)
        Non-null sample weights.

    X : ArrayLike of shape (n_samples, n_features)
        Training samples with non-null weights.

    y : ArrayLike of shape (n_samples,)
        Training labels with non-null weights.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.utils import check_null_weight
    >>> X = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y = np.array([5, 7, 9, 11, 13, 15])
    >>> sample_weight = np.array([0, 1, 1, 1, 1, 1])
    >>> sample_weight, X, y = check_null_weight(sample_weight, X, y)
    >>> print(sample_weight)
    [1. 1. 1. 1. 1.]
    >>> print(X)
    [[1]
     [2]
     [3]
     [4]
     [5]]
    >>> print(y)
    [ 7  9 11 13 15]
    """
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        non_null_weight = sample_weight != 0
        X = _safe_indexing(X, non_null_weight)
        y = _safe_indexing(y, non_null_weight)
        sample_weight = _safe_indexing(sample_weight, non_null_weight)
    sample_weight = cast(Optional[NDArray], sample_weight)
    return sample_weight, X, y


def fit_estimator(
    estimator: Union[RegressorMixin, ClassifierMixin],
    X: ArrayLike,
    y: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> Union[RegressorMixin, ClassifierMixin]:
    """
    Fit an estimator on training data by distinguishing two cases:
    - the estimator supports sample weights and sample weights are provided.
    - the estimator does not support samples weights or
      samples weights are not provided.

    Parameters
    ----------
    estimator : Union[RegressorMixin, ClassifierMixin]
        Estimator to train.

    X : ArrayLike of shape (n_samples, n_features)
        Input data.

    y : ArrayLike of shape (n_samples,)
        Input labels.

    sample_weight : Optional[ArrayLike] of shape (n_samples,)
        Sample weights. If None, then samples are equally weighted.
        By default None.

    Returns
    -------
    RegressorMixin
        Fitted estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.utils.validation import check_is_fitted
    >>> X = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y = np.array([5, 7, 9, 11, 13, 15])
    >>> estimator = LinearRegression()
    >>> estimator = fit_estimator(estimator, X, y)
    >>> check_is_fitted(estimator)
    """
    fit_parameters = signature(estimator.fit).parameters
    supports_sw = "sample_weight" in fit_parameters
    if supports_sw and sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def check_cv(
    cv: Optional[Union[int, str, BaseCrossValidator]] = None
) -> Union[str, BaseCrossValidator]:
    """
    Check if cross-validator is
    ``None``, ``int``, ``"prefit"`` or ``BaseCrossValidator``.
    Return a ``LeaveOneOut`` instance if integer equal to -1.
    Return a ``KFold`` instance if integer superior or equal to 2.
    Return a ``KFold`` instance if ``None``.
    Else raise error.

    Parameters
    ----------
    cv : Optional[Union[int, str, BaseCrossValidator]], optional
        Cross-validator to check, by default ``None``.

    Returns
    -------
    Optional[Union[float, str]]
        'prefit' or None.

    Raises
    ------
    ValueError
        If the cross-validator is not valid.
    """
    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, int):
        if cv == -1:
            return LeaveOneOut()
        if cv >= 2:
            return KFold(n_splits=cv)
    if isinstance(cv, BaseCrossValidator) or (cv == "prefit"):
        return cv
    raise ValueError(
        "Invalid cv argument. "
        "Allowed values are None, -1, int >= 2, 'prefit', "
        "or a BaseCrossValidator object (Kfold, LeaveOneOut)."
    )


def check_alpha(
    alpha: Optional[Union[float, Iterable[float]]] = None
) -> Optional[ArrayLike]:
    """
    Check alpha and prepare it as a ArrayLike.

    Parameters
    ----------
    alpha : Union[float, Iterable[float]]
        Can be a float, a list of floats, or a ArrayLike of floats.
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

    Returns
    -------
    ArrayLike
        Prepared alpha.

    Raises
    ------
    ValueError
        If alpha is not a float or an Iterable of floats between 0 and 1.

    Examples
    --------
    >>> from mapie.utils import check_alpha
    >>> check_alpha([0.5, 0.75, 0.9])
    array([0.5 , 0.75, 0.9 ])
    """
    if alpha is None:
        return alpha
    if isinstance(alpha, float):
        alpha_np = np.array([alpha])
    elif isinstance(alpha, Iterable):
        alpha_np = np.array(alpha)
    else:
        raise ValueError(
            "Invalid alpha. Allowed values are float or Iterable."
        )
    if len(alpha_np.shape) != 1:
        raise ValueError(
            "Invalid alpha. "
            "Please provide a one-dimensional list of values."
        )
    if alpha_np.dtype.type not in [np.float64, np.float32]:
        raise ValueError(
            "Invalid alpha. Allowed values are Iterable of floats."
        )
    if np.any(np.logical_or(alpha_np <= 0, alpha_np >= 1)):
        raise ValueError("Invalid alpha. Allowed values are between 0 and 1.")
    return alpha_np


def check_n_features_in(
    X: ArrayLike,
    cv: Optional[Union[float, str, BaseCrossValidator]] = None,
    estimator: Optional[Union[RegressorMixin, ClassifierMixin]] = None,
) -> int:
    """
    Check the expected number of training features.
    In general it is simply the number of columns in the data.
    If ``cv=="prefit"`` however,
    it can be deduced from the estimator's ``n_features_in_`` attribute.
    These two values absolutely must coincide.

    Parameters
    ----------
    cv : Optional[Union[float, str]]
        The cross-validation strategy for computing scores,
        by default ``None``.

    X : ArrayLike of shape (n_samples, n_features)
        Data passed into the ``fit`` method.

    estimator : RegressorMixin
        Backend estimator of MAPIE.

    Returns
    -------
    int
        Expected number of training features.

    Raises
    ------
    ValueError
        If there is an inconsistency between the shape of the dataset
        and the one expected by the estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.utils import check_n_features_in
    >>> X = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
    >>> print(check_n_features_in(X))
    5
    """
    if hasattr(X, "shape"):
        shape = np.shape(X)
        if len(shape) <= 1:
            n_features_in = 1
        else:
            n_features_in = shape[1]
    else:
        n_features_in = _num_features(X)
    if cv == "prefit" and hasattr(estimator, "n_features_in_"):
        if cast(Any, estimator).n_features_in_ != n_features_in:
            raise ValueError(
                "Invalid mismatch between "
                "X.shape and estimator.n_features_in_."
            )
    return n_features_in


def check_alpha_and_n_samples(
    alphas: Union[Iterable[float], float], n: int
) -> None:
    """
    Check if the quantile can be computed based
    on the number of samples and the alpha value.

    Parameters
    ----------
    alphas : Iterable[float]
        Iterable of floats.

    n : int
        number of samples.

    Raises
    ------
    ValueError
        If the number of samples of the score is too low,
        1/alpha (or 1/(1 - alpha)) must be lower than the number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.utils import check_alpha_and_n_samples
    >>> try:
    ...     check_alpha_and_n_samples(np.array([1,2,3]), 0.5)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Number of samples of the score is too low,
    1/alpha (or 1/(1 - alpha)) must be lower than the number of samples.
    """
    if isinstance(alphas, float):
        alphas = np.array([alphas])
    for alpha in alphas:
        if n < 1 / alpha or n < 1 / (1 - alpha):
            raise ValueError(
                "Number of samples of the score is too low,\n"
                "1/alpha (or 1/(1 - alpha)) must be lower "
                "than the number of samples."
            )


def check_n_jobs(n_jobs: Optional[int] = None) -> None:
    """
    Check parameter ``n_jobs``.

    Raises
    ------
    ValueError
        If parameter is not valid.

    Examples
    --------
    >>> from mapie.utils import check_n_jobs
    >>> try:
    ...     check_n_jobs(0)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid n_jobs argument. Must be different than 0.
    """
    if not isinstance(n_jobs, (int, type(None))):
        raise ValueError("Invalid n_jobs argument. Must be an integer.")

    if n_jobs == 0:
        raise ValueError("Invalid n_jobs argument. Must be different than 0.")


def check_verbose(verbose: int) -> None:
    """
    Check parameter ``verbose``.

    Raises
    ------
    ValueError
        If parameter is not valid.

    Examples
    --------
    >>> from mapie.utils import check_verbose
    >>> try:
    ...     check_verbose(-1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid verbose argument. Must be non-negative.
    """
    if not isinstance(verbose, int):
        raise ValueError("Invalid verbose argument. Must be an integer.")

    if verbose < 0:
        raise ValueError("Invalid verbose argument. Must be non-negative.")


def check_nan_in_aposteriori_prediction(X: ArrayLike) -> None:
    """
    Check that all the points are used at least once, otherwise this means
    you have set the number of subsamples too low.

    Parameters
    ----------
    X : Array of shape (size of training set, number of estimators) whose rows
    are the predictions by each estimator of each training sample.

    Raises
    ------
    Warning
        If the aggregated predictions of any training sample would be nan.
    Examples
    --------
    >>> import warnings
    >>> warnings.filterwarnings("error")
    >>> import numpy as np
    >>> from mapie.utils import check_nan_in_aposteriori_prediction
    >>> X = np.array([[1, 2, 3],[np.nan, np.nan, np.nan],[3, 4, 5]])
    >>> try:
    ...     check_nan_in_aposteriori_prediction(X)
    ... except Exception as exception:
    ...     print(exception)
    ...
    WARNING: at least one point of training set belongs to every resamplings.
    Increase the number of resamplings
    """
    if np.any(np.all(np.isnan(X), axis=1), axis=0):
        warnings.warn(
            "WARNING: at least one point of training set "
            + "belongs to every resamplings.\n"
            "Increase the number of resamplings"
        )


def check_lower_upper_bounds(
    y_preds: NDArray, y_pred_low: NDArray, y_pred_up: NDArray
) -> None:
    """
    Check if the lower or upper bounds are inconsistent.
    If checking for MapieQuantileRegressor, then also checking
    initial quantile values.

    Parameters
    ----------
    y_pred : NDArray of shape (n_samples, 3) or (n_samples,)
        All the predictions at quantile:
        alpha/2, (1 - alpha/2), 0.5 or only the predictions
    y_pred_low : NDArray of shape (n_samples,)
        Final lower bound prediction with additional quantile
        value added
    y_pred_up : NDArray of shape (n_samples,)
        Final upper bound prediction with additional quantile
        value added

    Raises
    ------
    Warning
        If the aggregated predictions of any training sample would be nan.

    Examples
    --------
    >>> import warnings
    >>> warnings.filterwarnings("error")
    >>> import numpy as np
    >>> from mapie.utils import check_lower_upper_bounds
    >>> y_preds = np.array([[4, 2, 3], [3, 4, 5], [2, 3, 4]])
    >>> y_pred_low = np.array([4, 3, 2])
    >>> y_pred_up = np.array([4, 4, 4])
    >>> try:
    ...     check_lower_upper_bounds(y_preds, y_pred_low, y_pred_up)
    ... except Exception as exception:
    ...     print(exception)
    ...
    WARNING: The initial prediction values from the quantile method
    present issues as the upper quantile values might be higher than the
    lower quantile values.
    """
    if y_preds.ndim == 1:
        init_pred = y_preds
    else:
        init_pred, init_lower_bound, init_upper_bound = y_preds
    if y_preds.ndim != 1 and np.any(
        np.logical_or(
            init_lower_bound >= init_upper_bound,
            init_pred <= init_lower_bound,
            init_pred >= init_upper_bound,
        )
    ):
        warnings.warn(
            "WARNING: The initial prediction values from the "
            + "quantile method\npresent issues as the upper "
            "quantile values might be higher than the\nlower "
            + "quantile values."
        )
    if np.any(
        np.logical_or(
            y_pred_low >= y_pred_up,
            init_pred <= y_pred_low,
            init_pred >= y_pred_up,
        )
    ):
        warnings.warn(
            "WARNING: Following the additional value added to have conformal "
            "predictions, the upper and lower bound present issues as one "
            "might be higher or lower than the other."
        )


def check_conformity_score(
    conformity_score: Optional[ConformityScore],
) -> ConformityScore:
    """
    Check parameter ``conformity_score``.

    Raises
    ------
    ValueError
        If parameter is not valid.

    Examples
    --------
    >>> from mapie.utils import check_conformity_score
    >>> try:
    ...     check_conformity_score(1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid conformity_score argument.
    Must be None or a ConformityScore instance.
    """
    if conformity_score is None:
        return AbsoluteConformityScore()
    elif isinstance(conformity_score, ConformityScore):
        return conformity_score
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a ConformityScore instance."
        )


def check_defined_variables_predict_cqr(
    ensemble: bool,
    alpha: Union[float, Iterable[float], None],
) -> None:
    """
    Check that the parameters defined for the predict method
    of ``MapieQuantileRegressor`` are correct.

    Parameters
    ----------
    ensemble : bool
        Ensemble has not been defined in predict and therefore should
        will not have any effects in this method.
    alpha : Optional[Union[float, Iterable[float]]]
        For ``MapieQuantileRegresor`` the alpha has to be defined
        directly in initial arguments of the class.

    Raises
    ------
    Warning
        If the ensemble value is defined in the predict function
        of ``MapieQuantileRegressor``.
    Warning
        If the alpha value is defined in the predict function
        of ``MapieQuantileRegressor``.

    Examples
    --------
    >>> import warnings
    >>> warnings.filterwarnings("error")
    >>> from mapie.utils import check_defined_variables_predict_cqr
    >>> try:
    ...     check_defined_variables_predict_cqr(True, None)
    ... except Exception as exception:
    ...     print(exception)
    ...
    WARNING: ensemble is not utilized in ``MapieQuantileRegressor``.
    """
    if ensemble is True:
        warnings.warn(
            "WARNING: ensemble is not utilized in ``MapieQuantileRegressor``."
        )
    if alpha is not None:
        warnings.warn(
            "WARNING: Alpha should not be specified in the prediction method\n"
            + "with conformalized quantile regression."
        )


def check_estimator_fit_predict(
    estimator: Union[RegressorMixin, ClassifierMixin]
) -> None:
    """
    Check that the estimator has a fit and precict method.

    Parameters
    ----------
    estimator : Union[RegressorMixin, ClassifierMixin]
        Estimator to train.

    Raises
    ------
    ValueError
        If the estimator does not have a fit or predict attribute.
    """
    if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
        raise ValueError(
            "Invalid estimator. "
            "Please provide a regressor with fit and predict methods."
        )


def check_alpha_and_last_axis(vector: NDArray, alpha_np: NDArray):
    """Check when the dimension of vector is 3 that its last axis
    size is the same than the number of alphas.

    Parameters
    ----------
    vector : NDArray of shape (n_samples, 1, n_alphas)
        Vector on which compute the quantile.
    alpha_np : NDArray of shape (n_alphas, )
        Confidence levels.


    Raises
    ------
    ValueError
        Error is the last axis dimension is different from the
        number of alphas.
    """
    if len(alpha_np) != vector.shape[2]:
        raise ValueError(
            "In case of the vector has 3 dimensions, the dimension\n"
            + "of his last axis must be equal to the number of alphas"
        )
    else:
        return vector, alpha_np


def compute_quantiles(vector: NDArray,
                      alpha: NDArray,
                      mondrian=False) -> NDArray:
    """Compute the desired quantiles of a vector.

    Parameters
    ----------
    vector : NDArray of shape Union[(n_samples, 1), (n_samples, 1, n_alphas)]
        Vector on which compute the quantile. If the vector has 3 dimensions,
        then each 1-alpha quantile will be computed on its corresping matrix
        selected on the last axis of the matrix.
    alpha : NDArray for shape (n_alphas, )
        Risk levels.

    Returns
    -------
    NDArray of shape (n_alphas, )
        Quantiles of the vector.
    """
    n = len(vector)
    if len(vector.shape) <= 2 and not mondrian:
        quantiles_ = np.stack([
                    np_quantile(
                        vector,
                        ((n + 1) * (1 - _alpha)) / n,
                        method="higher"
                    ) for _alpha in alpha
        ])
    elif len(vector.shape) <= 2 and mondrian:
        quantiles_ = np.stack([
                    np_quantile(
                        vector,
                        ((n + 1) * (1 - _alpha)) / n,
                        method="higher",
                        axis=0
                    ) for _alpha in alpha
        ])
    else:
        check_alpha_and_last_axis(vector, alpha)
        quantiles_ = np.stack(
                [
                    compute_quantiles(
                        vector[:, :, i],
                        np.array([alpha_])
                    ) for i, alpha_ in enumerate(alpha)
                ]
            )[:, 0]
    return quantiles_
