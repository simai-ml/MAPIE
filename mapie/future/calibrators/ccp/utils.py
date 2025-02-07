from __future__ import annotations

import inspect
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples

from mapie._typing import ArrayLike, NDArray


def format_functions(
    functions: Optional[Union[Callable, Iterable[Callable]]],
    bias: bool,
) -> List[Callable]:
    """
    Validate ``functions`` and add a column of ones, as a lambda function
    if ``bias=True``.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
        List of functions (or CCPCalibrator objects) or single function.

        Each function can take a combinaison of the following arguments:
        - ``X``: Input dataset, of shape (n_samples, ``n_in``)
        - ``y_pred``: estimator prediction, of shape (n_samples,)
        - ``z``: exogenous variable, of shape (n_samples, n_features).
            It should be given in the ``fit`` and ``predict`` methods.
        The results of each functions will be concatenated to build the final
        result of the transformation, of shape ``(n_samples, n_out)``, which
        will be used to estimate the conformity scores quantiles.

        If ``None``, return an empty list.

    bias: bool
        Whether or not to add a column of ones to the features.

    Returns
    -------
    List[Callable]
        ``functions`` as a not empty list

    Raises
    ------
    ValueError
        If ``functions`` is empty or ``None`` and ``bias=False``.
    """
    if functions is None:
        functions = []
    elif isinstance(functions, Iterable):
        functions = list(functions)
    else:
        functions = [functions]

    if bias:
        functions.append(lambda X: np.ones((_num_samples(X), 1)))
    if (len(functions) == 0):
        raise ValueError("You need to define the `functions` argument "
                         "with a function or a list of functions, "
                         "or keep bias argument to True.")
    return functions


def check_custom_calibrator_functions(
    functions: List[Callable]
) -> None:
    """
    Raise errors if the elements in ``functions`` have
    unexpected arguments.

    Raises
    ------
    ValueError
        If functions contain unknown required arguments.

    Notes
    -----
    This method ensures that the provided functions only use recognized
    arguments ('X', 'y_pred', 'z'). Unknown optional arguments are allowed,
    but will always use their default values.
    """
    error_ind: Dict[str, List[int]] = {}
    for i, funct in enumerate(functions):
        assert callable(funct)
        params = inspect.signature(funct).parameters

        for param, arg in params.items():
            if (
                param not in ["X", "y_pred", "z"]
                and arg.default is inspect.Parameter.empty
            ):
                if param in error_ind:
                    error_ind[param].append(i)
                else:
                    error_ind[param] = [i]

    if len(error_ind) > 0:
        error_msg = ""
        for param, inds in error_ind.items():
            error_msg += (
                f"The functions at index ({', '.join(map(str, inds))}) "
                + "of the 'functions' argument, has an unknown required "
                + f"argument '{param}'.\n"
            )
        raise ValueError(
            "Forbidden required argument in `CustomCCP` calibrator.\n"
            f"{error_msg}"
            "The only allowed required argument are : 'X', "
            "'y_pred' and 'z'.\n"
            "Note: You can use optional arguments if you want "
            "to. They will act as parameters, as it is always "
            "their default value which will be used."
        )


def sample_points(
    X: ArrayLike,
    points: Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]],
    multipliers: Optional[List[Callable]] = None,
) -> NDArray:
    """
    Generate the ``points_`` attribute from the ``points`` and ``X`` arguments.
    Only the samples which have weights (the value for each ``multipliers``
    function) different from ``0`` can be sampled.

    Parameters
    ----------
    X : ArrayLike
        Samples

    points : Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]
        If Array: List of data points, used as centers to compute
        gaussian distances. Should be an array of shape (n_points, n_in).

        If integer, the points will be sampled randomly from the ``X``
        dataset, where ``X`` is the data give to the
        ``GaussianCCP.fit`` method, which usually correspond to
        the ``X`` argument of the ``fit`` or ``fit_calibrator`` method
        of a ``SplitCP`` instance.

        You can pass a Tuple[ArrayLike, ArrayLike], to have a different
        ``sigma`` value for each point. The two elements of the
        tuple should be:
         - Data points: 2D array of shape (n_points, n_in)
         - Sigma values 2D array of shape (n_points, n_in) or (n_points, 1)
        In this case, the ``sigma``, ``random_sigma`` and ``X`` argument are
        ignored.

        If ``None``, default to ``20``.

    multipliers: Optional[List[Callable]]
        List of functions which should return an array of shape (n_samples, 1)
        or (n_samples, ) used to weight the sample.

    Returns
    -------
    NDArray
        2D NDArray of points

    Raises
    ------
    ValueError
        If ``points`` is an invalid argument.
    """
    if isinstance(points, int):
        if multipliers is None:
            not_null_index = list(range(_num_samples(X)))
        else:   # Only sample points which have a not null multiplier value
            test = np.ones((_num_samples(X), 1)).astype(bool)
            for f in multipliers:
                multi = f(X)
                if len(multi.shape) == 1:
                    multi = multi.reshape(-1, 1)
                test = test & (multi != 0)
            not_null_index = [i for i in range(_num_samples(X)) if test[i, 0]]
        if len(not_null_index) < points:
            if _num_samples(X) > points:
                raise ValueError("There are not enough samples with a "
                                 "multiplier value different from zero "
                                 f"to sample the {points} points.")
            else:
                raise ValueError("There is not enough valid samples from "
                                 f"which to sample the {points} points.")
        points_index = np.random.choice(
            not_null_index, size=points, replace=False
        )
        points_ = _safe_indexing(X, points_index)
    elif isinstance(points, tuple):
        points_ = np.array(points[0])
    elif len(np.array(points).shape) == 2:
        points_ = np.array(points)
    else:
        raise ValueError("Invalid `points` argument. The points argument"
                         "should be an integer, "
                         "a 2D array or a tuple of two 2D arrays.")
    return points_


def compute_sigma(
    X: ArrayLike,
    points: Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]],
    points_: NDArray,
    sigma: Optional[Union[float, ArrayLike]],
    random_sigma: bool,
    multipliers: Optional[List[Callable]] = None,

) -> NDArray:
    """
    Generate the ``sigmas_`` attribute from the ``points``, ``sigma``, ``X``
    arguments and the fitted ``points_``.

    Parameters
    ----------
    X : ArrayLike
        Samples

    points : Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]]
        Input ``points`` argument of ``GaussianCCP`` calibrator.

    points_ : NDArray
        Fitted 2D arrray of points

    sigma : Optional[Union[float, ArrayLike]]
        Standard deviation value used to compute the guassian distances,
        with the formula:
        ``np.exp(-0.5 * ((X - point) / sigma) ** 2)``
         - It can be an integer
         - It can be a 1D array of float with as many
        values as dimensions in the dataset

        If you want different standard deviation values of each points,
        you can indicate the sigma value of each point in the ``points``
        argument.

        If ``None``, ``sigma`` will default to a float equal to
        ``np.std(X)/(n**0.5)*d``
         - where ``X`` is the calibration data,
         passed to ``GaussianCCP.fit`` method, through
         ``SplitCPRegressor.fit/fit_calibrate`` method.
         - ``n`` is the number of points (``len(points)``).
         - ``d`` is the number of dimensions of ``X``.

    random_sigma : bool
        Whether to apply to the standard deviation values, a random multiplier,
        different for each point, equal to:

        ``2**np.random.normal(0, 1*2**(-2+np.log10(len(points))))``

        Exemple:
         - For 10 points, the sigma value will, in general,
        be multiplied by a value between 0.7 and 1.4
         - For 100 points, the sigma value will, in general,
        be multiplied by a value between 0.5 and 2

    multipliers: Optional[List[Callable]]
        List of functions which should return an array of shape (n_samples, 1)
        or (n_samples, ) used to weight the sample.

    Returns
    -------
    sigmas_
        2D NDArray of standard deviation values
    """
    # If each point has a corresponding sigma value
    if isinstance(points, tuple):
        sigmas_ = np.array(points[1], dtype=float)
        if len(sigmas_.shape) == 1:
            sigmas_ = sigmas_.reshape(-1, 1)
    # If sigma is not defined
    elif sigma is None:
        # We get the X indexes which correspond to a not zero multiplier value
        if multipliers is None:
            not_null_index = list(range(_num_samples(X)))
        else:
            test = np.ones((_num_samples(X), 1)).astype(bool)
            for f in multipliers:
                multi = f(X)
                if len(multi.shape) == 1:
                    multi = multi.reshape(-1, 1)
                test = test & (multi != 0)
            not_null_index = [i for i in range(_num_samples(X)) if test[i, 0]]

        points_std = np.std(_safe_indexing(X, not_null_index), axis=0)\
            / (_num_samples(points_)**0.5)\
            * _num_samples(_safe_indexing(X, 0))

        sigmas_ = np.ones((_num_samples(points_), 1))*points_std
    # If sigma is defined
    elif isinstance(points, int):
        sigmas_ = _init_sigmas(sigma, points)
    else:
        sigmas_ = _init_sigmas(sigma, _num_samples(points))

    if random_sigma:
        n = _num_samples(points_)
        sigmas_ *= (
            2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
            .reshape(-1, 1)
        )
    return cast(NDArray, sigmas_)


def _init_sigmas(
    sigma: Union[float, ArrayLike],
    n_points: int,
) -> NDArray:
    """
    If ``sigma`` is not ``None``, take a sigma value, and set ``sigmas_``
    to a standard deviation 2D array of shape (n_points, n_sigma),
    n_sigma being 1 or ``n_in``.

    Parameters
    ----------
    sigma : Union[float, ArrayLike]
        standard deviation, as float or 1D array of length ``n_in``
        (number of dimensins of the dataset)

    n_points : int
        Number of points user for gaussian distances calculation

    Raises
    ------
    ValueError
        If ``sigma`` is not None, a float or a 1D array
    """
    if isinstance(sigma, (float, int)):
        return np.ones((n_points, 1))*sigma
    else:
        if len(np.array(sigma).shape) != 1:
            raise ValueError("sigma argument should be a float "
                             "or a 1D array of floats.")
        return np.ones((n_points, 1))*np.array(sigma)


def dynamic_arguments_call(f: Callable, params_mapping: Dict) -> NDArray:
    """
    Call the function ``f``, with the correct arguments

    Parameters
    ----------
    f : Callable
        function to call

    params_mapping : Dict
        Dictionnary of argument names / values

    Returns
    -------
    NDArray
        result as 2D array
    """

    params = inspect.signature(f).parameters
    used_params = {
        p: params_mapping[p] for p in params
        if p in params_mapping and params_mapping[p] is not None
    }
    res = np.array(f(**used_params), dtype=float)
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=1)

    return res


def concatenate_functions(
    functions: List[Callable], params_mapping: Dict,
) -> NDArray:
    """
    Call the function of ``functions``, with the
    correct arguments, and concatenate the results, multiplied by each
    ``multipliers`` functions values.

    Parameters
    ----------
    functions : List[Callable]
        List of functions to call

    params_mapping : Dict
        Dictionnary of argument names / values

    Returns
    -------
    NDArray
        Concatenated result
    """
    # Compute phi(X, y_pred, z)
    result = np.hstack([
        dynamic_arguments_call(f, params_mapping) for f in functions
    ])
    return result


def check_multiplier(
    multipliers: Optional[List[Callable]],
    X: Optional[ArrayLike] = None,
    y_pred: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
) -> None:
    """
    Check if ``multipliers`` is a valid ``multiplier`` argument for
    ``CCPCalibrator``.

    Parameters
    ----------
    multipliers : List[Callable]
        function which sould return an array of shape (n_samples, 1) or
        (n_samples, )

    X : ArrayLike
            Observed samples

    y_pred : ArrayLike
        Target prediction

    z : ArrayLike
        Exogenous variable
    """
    if multipliers is None:
        return
    params_mapping = {"X": X, "y_pred": y_pred, "z": z}
    for f in multipliers:
        res = dynamic_arguments_call(f, params_mapping)
        if res.shape != (_num_samples(X), 1):
            raise ValueError("The function used as multiplier should return an"
                             "array of shape n_samples, 1) or (n_samples, ).\n"
                             f"Got shape = {res.shape}.")


def fast_mean_pinball_loss(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    sample_weight: Optional[NDArray] = None,
    alpha: float = 0.5
) -> float:
    """
    Pinball loss for quantile regression.
    It does the same as ``sklearn.metric.mean_minball_loss``, but without
    the checks on the ``y_true`` and ``y_pred`` arrays, for faster computation.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : NDArray of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : NDArray of shape (n_samples,), default=None
        Sample weights.

    alpha : float, slope of the pinball loss, default=0.5,
        This loss is equivalent to :ref:`mean_absolute_error` when `alpha=0.5`,
        `alpha=0.95` is minimized by estimators of the 95th percentile.

    Returns
    -------
    loss : float
        Weighted average of all output errors.
        The pinball loss output is a non-negative floating point. The best
        value is 0.0.
    """
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, weights=sample_weight, axis=0)

    return np.mean(output_errors)


def calibrator_optim_objective(
    beta: NDArray, calibrator_preds: NDArray, conformity_scores: NDArray,
    q: float, reg_param: Optional[float],
) -> float:
    """
    Objective funtcion to minimize to get the estimation of
    the conformity scores ``q`` quantile, caracterized by
    the scalar parameters in the ``beta`` vector.

    Parameters
    ----------
    beta : NDArray
        Parameters to optimize to minimize the objective function

    calibrator_preds : NDArray
        Transformation of the data X using the ``CCPCalibrator``.

    conformity_scores : NDArray
        Conformity scores of X

    q : float
        Between ``0.0`` and ``1.0``, represents the quantile, being
        ``1-alpha`` if ``alpha`` is the risk level of the confidence interval.

    reg_param: Optional[float]
        Float to monitor the ridge regularization
        strength. ``reg_param`` must be a non-negative
        float i.e. in ``[0, inf)``.

        .. warning::
            A too strong regularization may compromise the guaranteed
            marginal coverage. If ``calibrator.normalize=True``, it is usually
            recommanded to use ``reg_param < 1e-3``.

        If ``None``, no regularization is used.

        By default ``None``.

    Returns
    -------
    float
        Scalar value to minimize, being the sum of the pinball losses.
    """
    if reg_param is not None:
        reg_val = float(reg_param * np.linalg.norm(beta, ord=1))
    else:
        reg_val = 0
    return fast_mean_pinball_loss(
        y_true=conformity_scores, y_pred=calibrator_preds.dot(beta),
        alpha=q,
    ) + reg_val


def check_required_arguments(*args) -> None:
    """
    Make sure that the ``args`` arguments are not ``None``.

    It is used in calibrators based on ``BaseCalibrator``.
    Their ``fit`` and ``predict`` methods must have their custom
    arguments as optional (even the required ones), to match the base class
    signature. So we have to check that the required arguments
    are not ``None``.

    Raises
    ------
    ValueError
        If one of the passed argument is ``None``.
    """
    if any(arg is None for arg in args):
        raise ValueError("One of the required arguments is None."
                         "Fix the calibrator method definition.")
