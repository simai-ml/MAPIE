from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

from mapie._typing import ArrayLike

from .base import CCPCalibrator
from .utils import format_functions


class PolynomialCCP(CCPCalibrator):
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
    object with polynomial features of ``X``, ``y_pred`` or ``z``.

    See the examples and the documentation to build a
    :class:`~mapie.future.calibrators.ccp.CCPCalibrator`
    adaptated to your dataset and constraints.

    Parameters
    ----------
    degree: Union[int, List[int]]
        If ``degree`` is an integer, it correspond to the degree of the
        polynomial features transformer. It will create the features
        ``1``, ``variable``, ``variable``**2, ..., ``variable``**``degree``.

        If ``degree`` is a list of integers, it will create the features
        ``variable``**d, for all integer d in ``degree``

        ``variable`` may be ``X``, ``y_pred`` or ``z``, depending on the
        ``variable`` argument value.

        If ``None``, it will default to ``degree=1``.

        .. note::
            if ``0`` is in the considered exponents (if ``degree`` is an
            integer, or if ``0 in degree`` if it is a list), it is not
            ``variable**0`` of shape ``(n_samples, n_in)`` which is added,
            but only one feature of ones, of shape ``(n_samples, 1)``.
            It is actually equivalent to ``bias=True``.

        By default ``None``.

    variable: Literal["X", "y_pred", "z"]
        String, used to choose which argument between ``X``, ``y_pred`` and
        ``z`` is used to build the polynomial features.

        By default ``"X"``

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

        By default ``False``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

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

    exponents: List[int]
        List of exponents of the built polynomial features

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
    >>> from mapie.future.calibrators import PolynomialCCP
    >>> from mapie.future.split import SplitCPRegressor
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400, 2).reshape(-1, 1)
    >>> y_train = 1 + 2*X_train[:,0] + np.random.rand(len(X_train))
    >>> mapie = SplitCPRegressor(
    ...     calibrator=PolynomialCCP(1), alpha=0.1, random_state=1,
    ... ).fit(X_train, y_train)
    >>> y_pred, y_pi = mapie.predict(X_train)
    """
    def __init__(
        self,
        degree: Optional[Union[int, List[int]]] = None,
        variable: str = "X",
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        reg_param: Optional[float] = None,
    ) -> None:
        self.degree = degree
        self.variable = variable
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.reg_param = reg_param

        self._multipliers: Optional[List[Callable]] = None

    def _convert_degree(
        self, degree: Optional[Union[int, List[int]]], bias: bool
    ) -> Tuple[List[int], bool]:
        """
        Convert ``degree`` argument into a list of exponents

        Parameters
        ----------
        degree: Union[int, List[int]]
            If ``degree``is an integer, it correspond to the degree of the
            polynomial features. It will create the features
            ``1``, ``variable``, ``variable``**2, ...,
            ``variable``**``degree``.

            If ``degree``is an iterable of integers, it will create the
            features ``variable``**d, for all integer d in ``degree``

            ``variable`` may be ``X``, ``y_pred`` or ``z``, depending on the
            ``variable``argument value.

            If ``None``, it will default to ``degree=1``.

            By default ``None``.

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

        Returns
        -------
        Tuple[List[int], bool]
            - List of exponents (the exponent ``0`` will be replaced by
            ``bias=True``, which is equivalent. It is useless to add as many
            columns of ones as dimensions of ``X``. Only one is enough.)
            - new ``bias`` value.
        """
        if degree is None:
            exponents = [0, 1]
        elif isinstance(degree, int):
            exponents = list(range(degree+1))
        else:
            exponents = degree

        return exponents, (0 in exponents) or bias

    def _create_functions(
        self, exponents: List[int], variable: str
    ) -> List[Callable]:
        """
        Create the list of lambda functions, based on the list ``exponents``
        and the ``variable`` value.

        Parameters
        ----------
        exponents: List[int]
            List of exponents to apply on the ``variable```

        variable: Literal["X", "y_pred", "z"]
            Variable on which to apply the exponents.
        """
        if variable == "X":
            return [lambda X, d=d: X**d for d in exponents if d != 0]
        elif variable == "y_pred":
            return [lambda y_pred, d=d: y_pred**d for d in exponents if d != 0]
        elif variable == "z":
            return [lambda z, d=d: z**d for d in exponents if d != 0]
        else:
            raise ValueError("variable must be 'X', 'y_pred' or 'z'")

    def _check_transform_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Check the parameters required to call ``transform``.
        In particular, check that the ``functions``
        attribute is valid and set the ``functions_`` argument.

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
        self.exponents, self.bias = self._convert_degree(
            self.degree, self.bias)
        functions = self._create_functions(self.exponents, self.variable)
        self.functions_ = format_functions(functions, self.bias)
