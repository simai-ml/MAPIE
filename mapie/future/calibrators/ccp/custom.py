from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Union

from sklearn.utils import _safe_indexing

from mapie._typing import ArrayLike

from .base import CCPCalibrator
from .utils import (check_multiplier, check_custom_calibrator_functions,
                    format_functions)


class CustomCCP(CCPCalibrator):
    """
    Calibrator based on :class:`~mapie.future.calibrators.ccp.CCPCalibrator`,
    used in :class:`~mapie.future.split.SplitCPRegressor` or
    :class:`~mapie.future.split.SplitCPClassifier`
    to estimate the conformity scores.

    It corresponds to the adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in
    "Conformal Prediction With Conditional Guarantees" [1].

    The goal is to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

    This class builds a :class:`~mapie.future.calibrators.ccp.CCPCalibrator`
    object with custom features, function of ``X``, ``y_pred`` or ``z``,
    defined as a list of functions in ``functions`` argument.

    This class can be used to concatenate
    :class:`~mapie.future.calibrators.ccp.CCPCalibrator` instances.

    See the examples and the documentation to build a
    :class:`~mapie.future.calibrators.ccp.CCPCalibrator`
    adaptated to your dataset and constraints.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
        List of functions (or
        :class:`~mapie.future.calibrators.ccp.CCPCalibrator` objects)
        or single function.

        Each function can take a combinaison of the following arguments:

        - ``X``: Input dataset, of shape (n_samples, ``n_in``)
        - ``y_pred``: estimator prediction, of shape (n_samples,)
        - ``z``: exogenous variable, of shape (n_samples, n_features).
          It should be given in the ``fit`` and ``predict`` methods.

        The results of each functions will be concatenated to build the final
        result of the transformation, of shape ``(n_samples, n_out)``, which
        will be used to estimate the conformity scores quantiles.

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

    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up_[0]: Array of shape (calibrator.n_out, )
        beta_up_[1]: Whether the optimization process converged or not
                    (cover is not guaranteed if the optimisation has failed)

    beta_low_: Tuple[NDArray, bool]
        Same as ``beta_up_``, but for the lower bound

    References
    ----------
    [1]:
        Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
        "Conformal Prediction With Conditional Guarantees", 2023

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.future.calibrators import CustomCCP
    >>> from mapie.future.split import SplitCPRegressor
    >>> from mapie.conformity_scores import AbsoluteConformityScore
    >>> np.random.seed(1)
    >>> X_train = np.linspace(0, 3.14, 1001).reshape(-1, 1)
    >>> y_train = np.random.rand(len(X_train))*np.sin(X_train[:,0])
    >>> calibrator = CustomCCP(
    ...     functions=[
    ...         lambda X: np.sin(X[:,0]),
    ...     ],
    ...     bias=True,
    ... )
    >>> mapie = SplitCPRegressor(
    ...     calibrator=calibrator, alpha=0.1, random_state=1,
    ...     conformity_score=AbsoluteConformityScore(sym=False)
    ... ).fit(X_train, y_train)
    >>> y_pred, y_pi = mapie.predict(X_train)
    """
    transform_attributes: List[str] = ["functions_", "is_transform_fitted_"]

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable[Callable]]] = None,
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        reg_param: Optional[float] = None,
    ) -> None:
        super().__init__(functions, bias, normalized, init_value, reg_param)

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
        self.functions_ = format_functions(self.functions, self.bias)
        check_custom_calibrator_functions(self.functions_)

    def _transform_params(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> CustomCCP:
        """
        Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected array of features.

        It should set all the attributes of ``transform_attributes``
        (i.e. ``functions_``). It should also set, once fitted, ``n_in``,
        ``n_out`` and ``init_value_``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y_pred: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        check_multiplier(self._multipliers, X, y_pred, z)
        self._check_transform_parameters(X, y_pred, z)

        for phi in self.functions_:
            if isinstance(phi, CCPCalibrator):
                phi._transform_params(X, y_pred, z)
                check_multiplier(phi._multipliers, X, y_pred, z)
        self.is_transform_fitted_ = True

        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        return self
