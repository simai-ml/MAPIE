from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples

from mapie._typing import ArrayLike
from .base import CCPCalibrator

from .utils import compute_sigma, format_functions, sample_points


class GaussianCCP(CCPCalibrator):
    """
    Calibrator based on :class:`~mapie.future.calibrators.ccp.CCPCalibrator`,
    used in :class:`~mapie.future.split.SplitCPRegressor` or
    :class:`~mapie.future.split.SplitCPClassifier`
    to estimate the conformity scores.

    It corresponds to the adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".

    The goal is to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

    This class builds a :class:`~mapie.future.calibrators.ccp.CCPCalibrator`
    object with gaussian kernel features,
    which computes the gaussian distance between ``X`` and some points,
    randomly sampled in the dataset or set by the user.

    See the examples and the documentation to build a
    :class:`~mapie.future.calibrators.ccp.CCPCalibrator`
    adaptated to your dataset and constraints.

    Parameters
    ----------
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

        By default, ``20``

    sigma : Optional[Union[float, ArrayLike]]
        Standard deviation value used to compute the guassian distances,
        with the formula:
        np.exp(-0.5 * ((X - point) / ``sigma``) ** 2)
         - It can be an integer
         - It can be a 1D array of float with as many
        values as dimensions in the dataset

        If you want different standard deviation values of each points,
        you can indicate the sigma value of each point in the ``points``
        argument.

        If ``None``, ``sigma`` will default to a float equal to
        ``np.std(X)/(n**0.5)*d``
         - where ``X`` is the calibration data, passed to ``GaussianCCP.fit``
         method, which usually correspond to the ``X`` argument of the ``fit``
         or ``fit_calibrator`` method of a ``SplitCP`` instance.
         - ``n`` is the number of points used as gaussian centers.
         - ``d`` is the number of dimensions of ``X`` (i.e. ``n_in``).

        By default, ``None``

    random_sigma : bool
        Whether to apply to the standard deviation values, a random multiplier,
        different for each point, equal to:

        ``2**np.random.normal(0, 1*2**(-2+np.log10(len(points))))``

        Exemple:
         - For 10 points, the sigma value will be, in general,
         multiplied by a value between 0.7 and 1.4
         - For 100 points, the sigma value will be, in general,
         multiplied by a value between 0.5 and 2

        .. note::
            This is a default suggestion of randomization,
            which allow to have in the same time wide and narrow gaussians.

            You can use fully custom sigma values, buy passing to the
            ``points`` argument, a different sigma value for each point.

        By default, ``False``

    bias: bool
        Add a column of ones to the features,
        (to make sure that the marginal coverage is guaranteed).
        If the ``CCPCalibrator`` object definition covers all the dataset
        (meaning, for all calibration and test samples, the resulting
        ``calibrator.predict(X, y_pred, z)`` is never all zeros),
        this column of ones is not necessary to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        If you are not sure, use ``bias=True`` to garantee the marginal
        coverage.

        ..note::
            In this case, with ``GaussianCCP``, if ``normalized`` is
            ``True`` (it is, by default), the result of
            ``calibrator.predict(X, y_pred, z)``  will never
            be all zeros, so this ``bias`` is not required,
            to have a guaranteed coverage.

        By default ``False``.

    normalized: bool
        Whether or not to normalized the resulting
        ``calibrator.predict(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a width of zero for out-of-distribution samples.
        On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        .. note::
            To make sure that for too small ``sigma`` values,
            or for out-of-distribution samples, the interval width doesn't
            crash to zero, we set by default ``normalized = True``.
            By doing so, even the samples which were in any gaussian tild,
            will still be linked to the closest one.

        By default ``True``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, the initial vector is sampled from a normal distribution.

        By default ``None``.

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

    Attributes
    ----------
    transform_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``predict``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of ``calibrator.transform(X, y_pred, z)``

    points_: NDArray
        Array of shape (n_points, n_in), corresponding to the points used to
        compute the gaussian distanes.

    sigmas_: NDArray of shape (len(points), 1) or (len(points), n_in)
        Standard deviation values

    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up_[0]: Array of shape (calibrator.n_out, )
        beta_up_[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low_: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.future.calibrators import GaussianCCP
    >>> from mapie.future.split import SplitCPRegressor
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400, 2).reshape(-1, 1)
    >>> y_train = 1 + 2*X_train[:,0] + np.random.rand(len(X_train))
    >>> mapie = SplitCPRegressor(
    ...     calibrator=GaussianCCP(2), alpha=0.1, random_state=1,
    ... ).fit(X_train, y_train)
    >>> y_pred, y_pi = mapie.predict(X_train)
    """
    transform_attributes: List[str] = ["points_", "sigmas_", "functions_"]

    def __init__(
        self,
        points: Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]] = 20,
        sigma: Optional[Union[float, ArrayLike]] = None,
        random_sigma: bool = False,
        bias: bool = False,
        normalized: bool = True,
        init_value: Optional[ArrayLike] = None,
        reg_param: Optional[float] = None,
    ) -> None:
        self.points = points
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.reg_param = reg_param

        self._multipliers: Optional[List[Callable]] = None

    def _check_points_sigma(
        self, points: ArrayLike, sigmas: ArrayLike
    ) -> None:
        """
        Take 2D arrays of points and standard deviations and check
        compatibility

        Parameters
        ----------
        points : ArrayLike
            2D array of shape (n_points, n_in)
        sigmas : ArrayLike
            2D array of shape (n_points, 1) or (n_points, n_in)

        Raises
        ------
        ValueError
            - If ``points`` and ``sigmas`` don't have the same number of rows
            - If ``sigmas``is not of shape (n_points, 1) or (n_points, n_in)
        """
        if _num_samples(points) != _num_samples(sigmas):
            raise ValueError("There should have as many points as "
                             "standard deviation values")

        if len(_safe_indexing(sigmas, 0)) not in [
            1, len(_safe_indexing(points, 0))
        ]:
            raise ValueError("The standard deviation 2D array should be of "
                             "shape (n_points, 1) or (n_points, n_in).\n"
                             f"Got sigma of shape: ({_num_samples(sigmas)}, "
                             f"{len(_safe_indexing(points, 0))}).")

    def _check_transform_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Check the parameters required to call ``transform``.
        In particular, set the ``points_`` and ``sigmas_`` attributes, based
        on the ``points``, ``sigma`` and ``random_sigma`` arguments.
        Then, the ``functions_`` attributes is set, with functions to compute
        all the gaussian distances.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self.points_ = sample_points(X, self.points, self._multipliers)
        self.sigmas_ = compute_sigma(
            X, self.points, self.points_, self.sigma,
            self.random_sigma, self._multipliers
        )
        self._check_points_sigma(self.points_, self.sigmas_)

        functions = [
            lambda X, mu=_safe_indexing(self.points_, i),
            sigma=_safe_indexing(self.sigmas_, i):
            np.exp(-0.5 * np.sum(((X - mu) / sigma) ** 2, axis=1))
            for i in range(_num_samples(self.points_))
        ]
        self.functions_ = format_functions(functions, self.bias)
