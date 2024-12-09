import logging
import warnings
from inspect import signature
from typing import Any, Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (BaseCrossValidator, BaseShuffleSplit,
                                     KFold, LeaveOneOut, ShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (_check_sample_weight, _num_features,
                                      check_is_fitted, column_or_1d)

from ._compatibility import np_quantile
from ._typing import ArrayLike, NDArray

SPLIT_STRATEGIES = ["uniform", "quantile", "array split"]


def check_null_weight(
    sample_weight: Optional[ArrayLike], X: ArrayLike, y: ArrayLike
) -> Tuple[Optional[NDArray], ArrayLike, ArrayLike]:
    """
    Check sample weights and remove samples with null sample weights.

    Parameters
    ----------
    sample_weight: Optional[ArrayLike] of shape (n_samples,)
        Sample weights.
    X: ArrayLike of shape (n_samples, n_features)
        Training samples.
    y: ArrayLike of shape (n_samples,)
        Training labels.

    Returns
    -------
    sample_weight: Optional[NDArray] of shape (n_samples,)
        Non-null sample weights.

    X: ArrayLike of shape (n_samples, n_features)
        Training samples with non-null weights.

    y: ArrayLike of shape (n_samples,)
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
        sample_weight = cast(NDArray, sample_weight)
    return sample_weight, X, y


# TODO back-end: this will be useless in v1 because we'll not distinguish
# sample_weight from other fit_params
def fit_estimator(
    estimator: Union[RegressorMixin, ClassifierMixin],
    X: ArrayLike,
    y: ArrayLike,
    sample_weight: Optional[NDArray] = None,
    **fit_params,
) -> Union[RegressorMixin, ClassifierMixin]:
    """
    Fit an estimator on training data by distinguishing two cases:
    - the estimator supports sample weights and sample weights are provided.
    - the estimator does not support samples weights or
      samples weights are not provided.

    Parameters
    ----------
    estimator: Union[RegressorMixin, ClassifierMixin]
        Estimator to train.

    X: ArrayLike of shape (n_samples, n_features)
        Input data.

    y: ArrayLike of shape (n_samples,)
        Input labels.

    sample_weight : Optional[ArrayLike] of shape (n_samples,)
        Sample weights. If None, then samples are equally weighted.
        By default None.

    **fit_params : dict
            Additional fit parameters.

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
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


def check_cv(
    cv: Optional[Union[int, str, BaseCrossValidator, BaseShuffleSplit]] = None,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Union[str, BaseCrossValidator, BaseShuffleSplit]:
    """
    Check if cross-validator is
    ``None``, ``int``, ``"prefit"``, ``"split"``, ``BaseCrossValidator`` or
    ``BaseShuffleSplit``.
    Return a ``LeaveOneOut`` instance if integer equal to -1.
    Return a ``KFold`` instance if integer superior or equal to 2.
    Return a ``KFold`` instance if ``None``.
    Else raise error.

    Parameters
    ----------
    cv: Optional[Union[int, str, BaseCrossValidator, BaseShuffleSplit]]
        Cross-validator to check, by default ``None``.

    test_size: Optional[Union[int, float]]
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, it will be set to 0.1.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    random_state: Optional[Union[int, np.random.RandomState]], optional
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.
        By default ```None``.

    Returns
    -------
    Union[str, BaseCrossValidator, BaseShuffleSplit]
        The cast `cv` parameter.

    Raises
    ------
    ValueError
        If the cross-validator is not valid.
    """
    if random_state is None:
        random_seeds = cast(list, np.random.get_state())[1]
        random_state = np.random.choice(random_seeds)
    if cv is None:
        return KFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
    elif isinstance(cv, int):
        if cv == -1:
            return LeaveOneOut()
        elif cv >= 2:
            return KFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            raise ValueError(
                "Invalid cv argument. "
                "Allowed integer values are -1 or int >= 2."
            )
    elif isinstance(cv, BaseCrossValidator):
        return cv
    elif isinstance(cv, BaseShuffleSplit):
        return cv
    elif cv == "prefit":
        return cv
    elif cv == "split":
        return ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
    else:
        raise ValueError(
            "Invalid cv argument. "
            "Allowed values are -1, int >= 2, "
            "or a suitable BaseCrossValidator object"
        )


def check_no_agg_cv(
    X: ArrayLike,
    cv: Union[int, str, BaseCrossValidator, BaseShuffleSplit],
    no_agg_cv_array: list,
    y: Optional[ArrayLike] = None,
    groups: Optional[ArrayLike] = None
) -> bool:
    """
    Check if cross-validator is ``"prefit"``, ``"split"`` or any split
    equivalent `BaseCrossValidator` or `BaseShuffleSplit`.

    Parameters
    ----------
    X: ArrayLike of shape (n_samples, n_features)
        Training data.

    cv: Union[int, str, BaseCrossValidator, BaseShuffleSplit]
        Cross-validator to check.

    no_agg_cv_array: list
        List of all non-aggregated cv methods.

    y: Optional[ArrayLike] of shape (n_samples,)
        Input labels.

        By default ``None``.

    groups: Optional[ArrayLike] of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.

        By default ``None``.

    y: Optional[ArrayLike] of shape (n_samples,)
        Input labels.

        By default ``None``.

    groups: Optional[ArrayLike] of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.

        By default ``None``.

    Returns
    -------
    bool
        True if `cv` is a split equivalent / non-aggregated cv method.
    """
    if isinstance(cv, str):
        return cv in no_agg_cv_array
    elif isinstance(cv, int):
        return cv == 1
    elif hasattr(cv, "get_n_splits"):
        return cv.get_n_splits(X, y, groups) == 1
    else:
        raise ValueError(
            "Invalid cv argument. "
            "Allowed values must have the `get_n_splits` method "
            "with zero or one parameter (X)."
        )


def check_alpha(
    alpha: Optional[Union[float, Iterable[float]]] = None
) -> Optional[ArrayLike]:
    """
    Check alpha and prepare it as a ArrayLike.

    Parameters
    ----------
    alpha: Union[float, Iterable[float]]
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
            "Invalid confidence_level. Allowed values are float or Iterable."
        )
    if len(alpha_np.shape) != 1:
        raise ValueError(
            "Invalid confidence_level."
            "Please provide a one-dimensional list of values."
        )
    if alpha_np.dtype.type not in [np.float64, np.float32]:
        raise ValueError(
            "Invalid confidence_level. Allowed values are Iterable of floats."
        )
    if np.any(np.logical_or(alpha_np < 0, alpha_np > 1)):
        raise ValueError(
            "Invalid confidence_level. Allowed values are between 0 and 1."
        )
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
    cv: Optional[Union[float, str]]
        The cross-validation strategy for computing scores,
        by default ``None``.

    X: ArrayLike of shape (n_samples, n_features)
        Data passed into the ``fit`` method.

    estimator: RegressorMixin
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
                "Invalid mismatch between ",
                "X.shape and estimator.n_features_in_."
            )
    return n_features_in


def check_gamma(
    gamma: float
) -> None:
    """
    Check if gamma is between 0 and 1.

    Parameters
    ----------
    gamma: float

    Raises
    ------
    ValueError
        If gamma is lower than 0 or higher than 1.
    """
    if (gamma < 0) or (gamma > 1):
        raise ValueError(
            "Invalid gamma. Allowed values are between 0 and 1."
        )


def get_effective_calibration_samples(scores: NDArray, sym: bool):
    """
    Calculates the effective number of calibration samples.

    Parameters
    ----------
    scores: NDArray
        An array of scores.

    sym: bool
        A boolean indicating whether the scores are symmetric.

    Returns
    -------
    n: int
        The effective number of calibration samples.
    """
    n = np.sum(~np.isnan(scores))
    if not sym:
        n //= 2
    return n


def check_alpha_and_n_samples(
    alphas: Union[Iterable[float], float],
    n: int,
) -> None:
    """
    Check if the quantile can be computed based
    on the number of samples and the alpha value.

    Parameters
    ----------
    alphas: Iterable[float]
        Iterable of floats.

    n: int
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
    1/confidence_level and 1/(1 - confidence_level) must be
    lower than the number of samples.
    """
    if isinstance(alphas, float):
        alphas = np.array([alphas])
    for alpha in alphas:
        if n < np.max([1/alpha, 1/(1-alpha)]):
            raise ValueError(
                "Number of samples of the score is too low,\n"
                "1/confidence_level and 1/(1 - confidence_level) must be\n"
                "lower than the number of samples."
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
    X: Array of shape (size of training set, number of estimators) whose rows
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
    y_pred_low: NDArray,
    y_pred_up: NDArray,
    y_preds: NDArray
) -> None:
    y_pred_low = column_or_1d(y_pred_low)
    y_pred_up = column_or_1d(y_pred_up)
    y_preds = column_or_1d(y_preds)

    any_inversion = np.any(
        (y_pred_low > y_pred_up) |
        (y_preds < y_pred_low) |
        (y_preds > y_pred_up)
    )

    if any_inversion:
        initial_logger_level = logging.root.level
        logging.basicConfig(level=logging.INFO)
        logging.info(
            "The predictions are ill-sorted."
        )
        logging.basicConfig(level=initial_logger_level)


def check_defined_variables_predict_cqr(
    ensemble: bool,
    alpha: Union[float, Iterable[float], None],
) -> None:
    """
    Check that the parameters defined for the predict method
    of ``MapieQuantileRegressor`` are correct.

    Parameters
    ----------
    ensemble: bool
        Ensemble has not been defined in predict and therefore should
        will not have any effects in this method.
    alpha: Optional[Union[float, Iterable[float]]]
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
    estimator: Union[RegressorMixin, ClassifierMixin]
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
    vector: NDArray of shape (n_samples, 1, n_alphas)
        Vector on which compute the quantile.
    alpha_np: NDArray of shape (n_alphas, )
        Confidence levels.


    Raises
    ------
    ValueError
        Error is the last axis dimension is different from the
        number of alphas.
    """
    if len(alpha_np) != vector.shape[2]:
        raise ValueError(
            "In case of the vector has 3 dimensions, the dimension of its"
            + "last axis must be equal to the number of confidence levels"
        )
    else:
        return vector, alpha_np


def compute_quantiles(vector: NDArray, alpha: NDArray) -> NDArray:
    """Compute the desired quantiles of a vector.

    Parameters
    ----------
    vector: NDArray of shape Union[(n_samples, 1), (n_samples, 1, n_alphas)]
        Vector on which compute the quantile. If the vector has 3 dimensions,
        then each 1-alpha quantile will be computed on its corresping matrix
        selected on the last axis of the matrix.
    alpha: NDArray for shape (n_alphas, )
        Risk levels.

    Returns
    -------
    NDArray of shape (n_alphas, )
        Quantiles of the vector.
    """
    n = len(vector)
    if len(vector.shape) <= 2:
        quantiles_ = np.stack(
            [
                np_quantile(
                    vector,
                    ((n + 1) * (1 - _alpha)) / n,
                    method="higher",
                )
                for _alpha in alpha
            ]
        )

    else:
        check_alpha_and_last_axis(vector, alpha)
        quantiles_ = np.stack(
            [
                compute_quantiles(vector[:, :, i], np.array([alpha_]))
                for i, alpha_ in enumerate(alpha)
            ]
        )[:, 0]
    return quantiles_


def get_calib_set(
    X: ArrayLike,
    y: ArrayLike,
    sample_weight: Optional[NDArray] = None,
    calib_size: Optional[float] = 0.3,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    shuffle: Optional[bool] = True,
    stratify: Optional[ArrayLike] = None,
) -> Tuple[
    ArrayLike, ArrayLike, ArrayLike, ArrayLike,
    Optional[NDArray], Optional[NDArray]
]:
    """
    Split the dataset into training and calibration sets.

    Parameters
    ----------
    Same definition of parameters as for the ``fit`` method.

    Returns
    -------
    Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike,
        Optional[NDArray], Optional[NDArray]
    ]
    - [0]: ArrayLike of shape (n_samples_*(1-calib_size), n_features)
        X_train
    - [1]: ArrayLike of shape (n_samples_*(1-calib_size),)
        y_train
    - [2]: ArrayLike of shape (n_samples_*calib_size, n_features)
        X_calib
    - [3]: ArrayLike of shape (n_samples_*calib_size,)
        y_calib
    - [4]: Optional[NDArray] of shape (n_samples_*(1-calib_size),)
        sample_weight_train
    - [5]: Optional[NDArray] of shape (n_samples_*calib_size,)
        sample_weight_calib
    """
    if sample_weight is None:
        (
            X_train, X_calib, y_train, y_calib
        ) = train_test_split(
                X,
                y,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
        )
        sample_weight_train = sample_weight
        sample_weight_calib = None
    else:
        (
                X_train,
                X_calib,
                y_train,
                y_calib,
                sample_weight_train,
                sample_weight_calib,
        ) = train_test_split(
                X,
                y,
                sample_weight,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
        )
    X_train, X_calib = cast(ArrayLike, X_train), cast(ArrayLike, X_calib)
    y_train, y_calib = cast(ArrayLike, y_train), cast(ArrayLike, y_calib)
    return (
        X_train, y_train, X_calib, y_calib,
        sample_weight_train, sample_weight_calib
    )


def check_estimator_classification(
    X: ArrayLike,
    y: ArrayLike,
    cv: Union[str, BaseCrossValidator],
    estimator: Optional[ClassifierMixin],
) -> ClassifierMixin:
    """
    Check if estimator is ``None``,
    and returns a ``LogisticRegression`` instance if necessary.
    If the ``cv`` attribute is ``"prefit"``,
    check if estimator is indeed already fitted.
    Parameters
    ----------
    X: ArrayLike of shape (n_samples, n_features)
        Training data.
    y: ArrayLike of shape (n_samples,)
        Training labels.
    cv: Union[str, BaseCrossValidator]
        Cross validation parameter.
    estimator: Optional[ClassifierMixin]
        Estimator to check.
    Returns
    -------
    ClassifierMixin
        The estimator itself or a default ``LogisticRegression`` instance.
    Raises
    ------
    ValueError
        If the estimator is not ``None``
        and has no fit, predict, nor predict_proba methods.
    NotFittedError
        If the estimator is not fitted and ``cv`` attribute is "prefit".
    """
    if estimator is None:
        return LogisticRegression(multi_class="multinomial").fit(X, y)

    if isinstance(estimator, Pipeline):
        est = estimator[-1]
    else:
        est = estimator
    if (
        not hasattr(est, "fit")
        and not hasattr(est, "predict")
        and not hasattr(est, "predict_proba")
    ):
        raise ValueError(
            "Invalid estimator. "
            "Please provide a classifier with fit,"
            "predict, and predict_proba methods."
        )
    if cv == "prefit":
        check_is_fitted(est)
        if not hasattr(est, "classes_"):
            raise AttributeError(
                "Invalid classifier. "
                "Fitted classifier does not contain "
                "'classes_' attribute."
            )
    return estimator


def get_binning_groups(
    y_score: NDArray,
    num_bins: int,
    strategy: str,
) -> NDArray:
    """
    Parameters
    ----------
    y_score : NDArray of shape (n_samples,)
        The scores given from the calibrator.
    num_bins : int
        Number of bins to make the split in the y_score.
    strategy : string
        The splitting strategy to split y_scores into different bins.
    Returns
    -------
    NDArray of shape (num_bins,)
        An array of all the splitting points for a new bin.
    """
    bins = None
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, num_bins)
        bins = np.percentile(y_score, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, num_bins)
    else:
        bin_groups = np.array_split(y_score, num_bins)
        bins = np.sort(np.array(
                [
                    bin_group.max() for bin_group in bin_groups[:-1]
                ]
                + [np.inf]
            )
        )
    return bins


def calc_bins(
    y_true: NDArray,
    y_score: NDArray,
    num_bins: int,
    strategy: str,
) -> Union[NDArray, NDArray, NDArray, NDArray]:
    """
    For each bins, calculate the accuracy, average confidence and size.
    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        The "true" values, target for the calibrator.
    y_score: NDArray of shape (n_samples,)
        The scores given from the calibrator.
    num_bins: int
        Number of bins to make the split in the y_score.
    strategy: str
        The way of splitting the predictions into different bins.
    Returns
    -------
    Union[NDArray, NDArray, NDArray, NDArray]
    - [0]: NDArray of shape (num_bins,)
    An array of all the splitting points for a new bin.
    - [1]: NDArray of shape (num_bins,)
    An array of the average accuracy in each of the bins.
    - [2]: NDArray of shape (num_bins,)
    An array of the average confidence in each of the bins.
    - [3]: NDArray of shape (num_bins,)
    An array of the number of observations in each of the bins.
    """
    bins = get_binning_groups(y_score, num_bins, strategy)
    binned = np.digitize(y_score, bins, right=True)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(y_score[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = np.divide(
                np.sum(y_true[binned == bin]),
                bin_sizes[bin],
            )
            bin_confs[bin] = np.divide(
                np.sum(y_score[binned == bin]),
                bin_sizes[bin],
            )
    return bins, bin_accs, bin_confs, bin_sizes  # type: ignore


def check_split_strategy(
    strategy: Optional[str]
) -> str:
    """
    Checks that the split strategy provided is valid
    and defults None split strategy to "uniform".
    Parameters
    ----------
    strategy: Optional[str]
        Can be a string or None.

    Returns
    -------
    str
        The spitting strategy that will be adopted, needs to be a string.

    Raises
    ------
    ValueError
        If the strategy is not part of the valid strategies.
    """
    if strategy is None:
        strategy = "uniform"
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            "Please provide a valid splitting strategy."
        )
    return strategy


def check_number_bins(
    num_bins: int
) -> int:
    """
    Checks that the bin specified is a number.

    Parameters
    ----------
    num_bins: int
        An integer that determines the number of bins to create
        on an array.

    Raises
    ------
    ValueError
        When num_bins is not an integer is raises an error.

    ValueError
        When num_bins is a negative number is raises an error.
    """
    if isinstance(num_bins, int) is False:
        raise ValueError(
            "Please provide a bin number as an integer."
        )
    elif num_bins < 1:
        raise ValueError(
            """
            Please provide a bin number greater than
            or equal to  1.
            """
        )
    else:
        return num_bins


def check_binary_zero_one(
    y_true: ArrayLike
) -> NDArray:
    """
    Checks if the array is binary and changes a non binary array
    to a zero, one array.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        Could be any array, but in this case is the true values
        as binary input.

    Returns
    -------
    NDArray of shape (n_samples,)
        An array of zero, one values.

    Raises
    ------
    ValueError
        If the input array is not binary, then an error is raised.
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    if type_of_target(y_true) == "binary":
        if ((np.unique(y_true) != np.array([0, 1])).any() and
                len(np.unique(y_true)) == 2):
            idx_min = np.where(y_true == np.min(y_true))[0]
            y_true[idx_min] = 0
            idx_max = np.where(y_true == np.max(y_true))[0]
            y_true[idx_max] = 1
            return y_true
        else:
            return y_true
    else:
        raise ValueError(
            "Please provide y_true as a binary array."
        )


def fix_number_of_classes(
    n_classes_: int,
    n_classes_training: NDArray,
    y_proba: NDArray
) -> NDArray:
    """
    Fix shape of y_proba of validation set if number of classes
    of the training set used for cross-validation is different than
    number of classes of the original dataset y.

    Parameters
    ----------
    n_classes_training: NDArray
        Classes of the training set.
    y_proba: NDArray
        Probabilities of the validation set.

    Returns
    -------
    NDArray
        Probabilities with the right number of classes.
    """
    y_pred_full = np.zeros(
        shape=(len(y_proba), n_classes_)
    )
    y_index = np.tile(n_classes_training, (len(y_proba), 1))
    np.put_along_axis(
        y_pred_full,
        y_index,
        y_proba,
        axis=1
    )
    return y_pred_full


def check_array_shape_classification(
    y_true: NDArray,
    y_pred_set: NDArray
) -> NDArray:
    """
    Fix shape of y_pred_set (to 3d array of shape (n_obs, n_class, n_alpha)).

    Parameters
    ----------
    y_true: ArrayLike
        True labels.
    y_pred_set: ArrayLike
        Prediction sets given by booleans of labels.

    Returns
    -------
    NDArray
        Fixed y_pred_set.

    Raises
    ------
    ValueError
        If y_true and y_pred_set doesn't have the same number of samples
        and if y_pred_sets is an array of shape greater than 3 or lower than 2.
    """
    if y_true.shape[0] != y_pred_set.shape[0]:
        raise ValueError(
            f"shape mismatch between y_true {y_true.shape} \
                and y_pred_set {y_pred_set.shape}"
        )
    if len(y_pred_set.shape) != 3:
        if len(y_pred_set.shape) != 2:
            raise ValueError(
                "y_pred_set should be a 3D array of shape \
                (n_obs, n_classes, n_confidence_levels)"
            )
        else:
            y_pred_set = np.expand_dims(y_pred_set, axis=2)
    return y_pred_set


def check_array_shape_regression(
    y_true: NDArray,
    y_intervals: NDArray
) -> NDArray:
    """
    Fix shape of y_intervals (to 3d array of shape (n_obs, 2, n_alpha)).

    Parameters
    ----------
    y_true: NDArray
        True labels.
    y_intervals: NDArray
        Lower and upper bound of prediction intervals
        with different alpha risks.

    Returns
    -------
    NDArray
        Fixed y_intervals.

    Raises
    ------
    ValueError
        If y_true and y_intervals doesn't have the same number of samples
        and if y_intervals is an array of shape greater than 3 or lower than 2.
    """
    if len(y_intervals.shape) != 3:
        if len(y_intervals.shape) != 2:
            raise ValueError(
                "y_intervals should be a 3D array of shape"
                " (n_obs, 2, n_confidence_levels)"
            )
        else:
            y_intervals = np.expand_dims(y_intervals, axis=2)
    if y_true.shape[0] != y_intervals.shape[0]:
        raise ValueError(
            f"shape mismatch between y_true {y_true.shape} \
                and y_intervals {y_intervals.shape}"
        )
    return y_intervals


def check_nb_intervals_sizes(widths: NDArray, num_bins: int) -> None:
    """
    Checks that the number of bins is less than the number of different
    interval widths.

    Parameters
    ----------
    widths: NDArray (n_samples, n_alpha)
        Widths of the prediction intervals.
    num_bins: int
        Number of bins.

    Raises
    ------
    ValueError
        If the number of bins is greater than the number of different widths.
    """
    for alpha in range(widths.shape[1]):
        nb_widths = len(np.unique(widths[:, alpha].round(5)))
        if nb_widths <= num_bins:
            raise ValueError(
                "The number of bins should be lower or equal to the number of \
                different interval widths."
            )


def check_nb_sets_sizes(sizes: NDArray, num_bins: int) -> None:
    """
    Checks that the number of bins is less than the number of different
    set sizes.

    Parameters
    ----------
    sizes: NDArrat of shape (n_samples, n_alpha)
        Sizes of the prediction sets.
    num_bins: int
        Number of bins.

    Raises
    ------
    ValueError
        If the number of bins is greater than the number of different sizes.
    """
    for alpha in range(sizes.shape[1]):
        nb_sizes = len(np.unique(sizes[:, alpha]))
        if nb_sizes <= num_bins:
            raise ValueError(
                "The number of bins should be less than the number of \
                different set sizes."
            )


def check_array_nan(array: NDArray) -> None:
    """
    Checks if the array have only NaN values. If it has we throw an error.

    Parameters
    ----------
    array: NDArray
        an array with non-numerical or non-categorical values

    Raises
    ------
    ValueError
        If all elements of the array are NaNs
    """
    if np.isnan(array).all() and len(np.unique(array)) > 0:
        raise ValueError(
            "Array contains only NaN values."
        )


def check_array_inf(array: NDArray) -> None:
    """
    Checks if the array have inf.
    If a value is infinite, we throw an error.

    Parameters
    ----------
    array: NDArray
        an array with non-numerical or non-categorical values

    Raises
    ------
    ValueError
        If any elements of the array is +inf or -inf.
    """
    if np.isinf(array).any():
        raise ValueError(
            "Array contains infinite values."
        )


def check_arrays_length(*arrays: NDArray) -> None:
    """
    Checks if the length of all arrays given in this function are the same

    Parameters
    ----------
    *arrays: NDArray
        Arrays expected to have the same length

    Raises
    ------
    ValueError
        If the length of the arrays are different
    """
    res = [array.shape[0] for array in arrays]
    if len(np.unique(res)) > 1:
        raise ValueError(
                "There are arrays with different length"
            )


def check_n_samples(
    X: NDArray,
    n_samples: Optional[Union[float, int]],
    indices: NDArray
) -> int:
    """
    Check alpha and prepare it as a ArrayLike.

    Parameters
    ----------
    n_samples: Union[float, int]
        Can be a float between 0 and 1 or a int
        Between 0 and 1, represent the part of data in the train sample
        When n_samples is a int, it represents the number of elements
        in the train sample

    Returns
    -------
    int
        n_samples

    Raises
    ------
    ValueError
        If n_samples is not an int in the range [1, inf)
        or a float in the range (0.0, 1.0)
    """
    if n_samples is None:
        n_samples = len(indices)
    elif isinstance(n_samples, float):
        if 0 < n_samples < 1:
            n_samples = int(np.floor(n_samples * X.shape[0]))
            if n_samples == 0:
                raise ValueError(
                    "The value of n_samples is too small. "
                    "You need to increase it so that n_samples*X.shape[0] > 1"
                    "otherwise n_samples should be an int"
                    )
        else:
            raise ValueError(
                "Invalid n_samples. Allowed values "
                "are float in the range (0.0, 1.0) or"
                " int in the range [1, inf)"
                )
    elif isinstance(n_samples, int) and n_samples <= 0:
        raise ValueError(
             "Invalid n_samples. Allowed values "
             "are float in the range (0.0, 1.0) or"
             " int in the range [1, inf)"
             )
    return int(n_samples)


def check_predict_params(
    predict_params_used_in_fit: bool,
    predict_params: dict,
    cv: Optional[Union[int, str, BaseCrossValidator]] = None
) -> None:
    """
    Check that if predict_params is used in the predict method,
    it is also used in the fit method. Otherwise, raise an error.
    Parameters
    ----------
    predict_params_used_in_fit: bool
        True if one or more predict_params are used in the fit method

    predict_param: dict
        Contains all predict params used in predict method

    Raises
    ------
    ValueError
        If any predict_params are used in the predict method but none
        are used in the fit method.
    """
    if cv != "prefit":
        if len(predict_params) > 0 and predict_params_used_in_fit is False:
            raise ValueError(
                f"Using 'predict_params' '{predict_params}' "
                f"without using one 'predict_params' in the fit method. "
                f"Please ensure a similar configuration of 'predict_params' "
                f"is used in the fit method before calling it in predict."
            )
        if len(predict_params) == 0 and predict_params_used_in_fit is True:
            raise ValueError(
                "Using one 'predict_params' in the fit method "
                "without using one 'predict_params' in the predict method. "
                "Please ensure a similar configuration of 'predict_params' "
                "is used in the predict method as called in the fit."
            )
