from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.future.calibrators.base import BaseCalibrator
from mapie.future.calibrators.ccp.utils import (
    calibrator_optim_objective, check_multiplier,
    check_custom_calibrator_functions, concatenate_functions,
    check_required_arguments, dynamic_arguments_call
)


class CCPCalibrator(BaseCalibrator, metaclass=ABCMeta):
    """
    Base abstract class for the calibrators used in
    :class:`~mapie.future.split.SplitCPRegressor` or
    :class:`~mapie.future.split.SplitCPClassifier`
    to estimate the conformity scores.
    It corresponds to the adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".

    The goal is to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

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
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023
    """
    transform_attributes: List[str] = ["functions_"]
    fit_attributes: List[str] = ["beta_up_", "beta_low_"]

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable[Callable]]] = None,
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        reg_param: Optional[float] = None,
    ) -> None:
        self.functions = functions
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.reg_param = reg_param

        self._multipliers: Optional[List[Callable]] = None

    @abstractmethod
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

    def _check_init_value(
        self, init_value: Optional[ArrayLike], n_out: int
    ) -> ArrayLike:
        """
        Set the ``init_value_`` attribute depending on ``init_value`` argument.
        If ``init_value = None``, ``init_value_`` is set to
        ``np.random.normal(0, 1, n_out)``.

        Parameters
        ----------
        init_value : Optional[ArrayLike]
            Optimization initialisation value, set at ``CCPCalibrator``
            initialisation.

        n_out : int
            Number of dimensions of the ``CCPCalibrator`` transformation.

        Returns
        -------
        ArrayLike
            Optimization initialisation value
        """
        if init_value is None:
            return np.random.normal(0, 1, n_out)
        else:
            return init_value

    def _check_optimization_success(
        self, *optimization_results: OptimizeResult
    ) -> None:
        """
        Check that all the ``optimization_results`` have successfully
        converged.

        Parameters
        ----------
        *optimization_resutls: OptimizeResult
            Scipy optimization outputs
        """
        for res in optimization_results:
            if not res.success:
                warnings.warn(
                    "WARNING: The optimization process "
                    f"failed with the following error: \n"
                    f"{res.message}\n"
                    "The returned prediction interval may be inaccurate."
                )

    def _transform_params(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> CCPCalibrator:
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
        # Fit the calibrator
        self._check_transform_parameters(X, y_pred, z)
        # Do some checks
        check_multiplier(self._multipliers, X, y_pred, z)
        result = self.transform(X, y_pred, z)

        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        return self

    def fit(
        self,
        X_calib: ArrayLike,
        conformity_scores_calib: NDArray,
        y_pred_calib: Optional[ArrayLike] = None,
        z_calib: Optional[ArrayLike] = None,
        **optim_kwargs,
    ) -> CCPCalibrator:
        """
        Fit the calibrator. It should set all the ``transform_attributes``
        and ``fit_attributes``.

        Parameters
        ----------
        X_calib: ArrayLike of shape (n_samples, n_features)
            Calibration data with not-null weights.

        conformity_scores_calib: ArrayLike of shape (n_samples,)
            Calibration conformity scores with not-null weights.

        y_pred_calib: ArrayLike of shape (n_samples,)
            Calibration target with not-null weights.

        z_calib: Optional[ArrayLike] of shape
        (n_calib_samples, n_exog_features)
            Exogenous variables with not-null weights.

            By default ``None``.

        optim_kwargs: Dict
            Other argument, used in sklear.optimize.minimize.
            Can be any of : ``method, jac, hess, hessp, bounds, constraints,
            tol, callback, options``

            By default, we use ``method='SLSQP'`` and
            ``options={'maxiter: 1000}``.
        """
        check_required_arguments(self.alpha)
        self.alpha = cast(float, self.alpha)

        if self.sym:
            q = 1 - self.alpha
        else:
            q = 1 - self.alpha / 2

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._transform_params(X_calib, y_pred_calib, z_calib)

        cs_features = self.transform(X_calib, y_pred_calib, z_calib)

        self._check_unconsistent_features(cs_features)

        not_nan_index = np.where(~np.isnan(conformity_scores_calib))[0]
        # Some conf. score values may be nan (ex: with ResidualNormalisedScore)

        if "method" not in optim_kwargs:
            optim_kwargs["method"] = "SLSQP"
        if "options" not in optim_kwargs:
            optim_kwargs["options"] = {}
        if "maxiter" not in optim_kwargs["options"]:
            optim_kwargs["options"]["maxiter"] = 1000

        self.calib_cs_features = cs_features[not_nan_index, :]
        self.conformity_scores_calib = conformity_scores_calib[not_nan_index]
        self.q = q
        self.reg_param

        self.optim_kwargs = optim_kwargs

        optimal_beta_up = cast(OptimizeResult, minimize(
            calibrator_optim_objective, self.init_value_,
            args=(
                cs_features[not_nan_index, :],
                conformity_scores_calib[not_nan_index],
                q,
                self.reg_param,
                ),
            **optim_kwargs,
        ))

        if not self.sym:
            optimal_beta_low = cast(OptimizeResult, minimize(
                calibrator_optim_objective, self.init_value_,
                args=(
                    cs_features[not_nan_index, :],
                    -conformity_scores_calib[not_nan_index],
                    q,
                    self.reg_param,
                ),
                **optim_kwargs,
            ))
        else:
            optimal_beta_low = optimal_beta_up

        self._check_optimization_success(optimal_beta_up, optimal_beta_low)

        self.beta_up_ = cast(Tuple[NDArray, bool],
                             (optimal_beta_up.x, optimal_beta_up.success))
        self.beta_low_ = cast(Tuple[NDArray, bool],
                              (optimal_beta_low.x,
                               optimal_beta_low.success))

        return self

    def _check_unconsistent_features(self, cs_features: NDArray) -> None:
        """
        Check if the ``cs_features`` array has rows full of zeros.
        """
        if np.any(np.all(cs_features == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "calibrator.transform(X, y_pred, z) is full of "
                          "zeros. It will result in a prediction interval of "
                          "zero width. Consider changing the `CCPCalibrator` "
                          "definintion.\nFix: Use `bias=True` "
                          "in the `CCPCalibrator` definition.")

    def transform(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Transform ``(X, y_pred, z)`` into an array of shape
        ``(n_samples, n_out)`` which represent features to estimate the
        conformity scores.

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : ArrayLike
            Target prediction

        z : ArrayLike
            Exogenous variable

        Returns
        -------
        NDArray
            features
        """
        check_is_fitted(self, self.transform_attributes)

        params_mapping = {"X": X, "y_pred": y_pred, "z": z}
        cs_features = concatenate_functions(self.functions_, params_mapping)

        if self.normalized:
            norm = cast(NDArray,
                        np.linalg.norm(cs_features, axis=1)).reshape(-1, 1)
            # the rows full of zeros are replace by rows of ones
            cs_features[(abs(norm) == 0)[:, 0], :] = np.ones(
                cs_features.shape[1])
            norm[abs(norm) == 0] = 1
            cs_features /= norm

        # Multiply the result by each multiplier function
        if self._multipliers is not None:
            for f in self._multipliers:
                cs_features *= dynamic_arguments_call(f, params_mapping)

        return cs_features

    def _get_cs_bound(
        self,
        conformity_scores: NDArray,
    ) -> Tuple[float, float]:
        """
        Create a valid up and down conformity score bound, based on
        ``cs_bound``

        Parameters
        ----------
        cs_bound: Optional[Union[float, Tuple[float, float]]]
            Bound of the conformity scores, such as for all conformity score S
            corresponding to ``X`` and ``y_pred``:

             - If the conformity score has ``sym=True``:
               ``cs_bound`` is a ``float`` and ``|S| <= cs_bound``

             - If the conformity score has ``sym=False``:
               ``cs_bound`` is a ``Tuple[float, float]`` and
               ``cs_bound[0] <= S <= cs_bound[1]``

            If ``cs_bound=None``,
            the maximum (and minimum if ``sym=False``) value
            of the calibration conformity scores is used.

            By default ``None``

        sym : bool
            Whether or not the computed prediction intervals should be
            symetrical or not

        conformity_scores: NDArray
            Conformity scores, used to estimate the bounds if ``cs_bound=None``

        Returns
        -------
        Tuple[float, float]
            (cs_bound_up, cs_bound_low)
        """

        cs_bound_up = max(conformity_scores)
        cs_bound_low = min(conformity_scores)

        return cs_bound_up, cs_bound_low

    def predict(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        unsafe_approximation: bool = False,
        **kwargs,
    ) -> NDArray:
        """
        Predict the conformity scores estimation by:
         - Transforming ``(X, y_pred, z)`` into an array of features of shape
         ``(n_samples, n_out)``
         - computing the dot product with the optimized beta values.

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : Optional[ArrayLike]
            Target prediction

        z : Optional[ArrayLike]
            Exogenous variable

        cs_bound: Optional[Union[float, Tuple[float, float]]]
            Bound of the conformity scores, such as for all conformity score S
            corresponding to ``X`` and ``y_pred``:

             - If the conformity score has ``sym=True``:
               ``cs_bound`` is a ``float`` and ``|S| <= cs_bound``

             - If the conformity score has ``sym=False``:
               ``cs_bound`` is a ``Tuple[float, float]`` and
               ``cs_bound[0] <= S <= cs_bound[1]``

            If ``cs_bound=None``,
            the maximum (and minimum if ``sym=False``) value
            of the calibration conformity scores is used.

            By default ``None``

        unsafe_approximation: Bool
            The most of the computation is done during the calibration phase
            (``fit`` method).
            However, the theoretical guarantees of the method rely on a small
            adjustment of the calibration for each test point. It will induce
            a conservatice interval prediction (potentially with over-coverage)
            and a long inference time, depending on the numbere of test points.

            Using ``unsafe_approximation = True`` will desactivate this
            correction, providing the interval predictions almost instantly.
            However, it can result in a small miss-coverage, as the previous
            guarantees don't hold anymore.

            By default, ``False``

        Returns
        -------
        NDArray
            Transformation
        """
        check_required_arguments(y_pred)

        check_is_fitted(self, self.transform_attributes + self.fit_attributes)

        cs_features = self.transform(X, y_pred, z)

        self._check_unconsistent_features(cs_features)

        if unsafe_approximation:
            y_pred_low = -cs_features.dot(self.beta_low_[0][:, np.newaxis])
            y_pred_up = cs_features.dot(self.beta_up_[0][:, np.newaxis])
        else:
            cs_bound_up, cs_bound_low = self._get_cs_bound(
                self.conformity_scores_calib
            )

            y_pred_up = np.zeros((_num_samples(X), 1))
            y_pred_low = np.zeros((_num_samples(X), 1))
            for i in range(len(y_pred_up)):
                corrected_beta_up = cast(OptimizeResult, minimize(
                    calibrator_optim_objective, self.beta_up_[0],
                    args=(
                        np.vstack(
                            [self.calib_cs_features, cs_features[[i], :]]
                        ),
                        np.hstack(
                            [self.conformity_scores_calib, [cs_bound_up]]
                        ),
                        self.q,
                        self.reg_param,
                    ),
                    **self.optim_kwargs,
                ))

                if not self.sym:
                    corrected_beta_low = cast(OptimizeResult, minimize(
                        calibrator_optim_objective, self.beta_low_[0],
                        args=(
                            np.vstack(
                                [self.calib_cs_features, cs_features[[i], :]]
                            ),
                            -np.hstack(
                                [self.conformity_scores_calib, [cs_bound_low]]
                            ),
                            self.q,
                            self.reg_param,
                        ),
                        **self.optim_kwargs,
                    ))

                else:
                    corrected_beta_low = corrected_beta_up

                self._check_optimization_success(
                    corrected_beta_up, corrected_beta_low
                )

                y_pred_up[[i]] = cs_features[[i], :].dot(
                    corrected_beta_up.x[:, np.newaxis]
                )
                y_pred_low[[i]] = -cs_features[[i], :].dot(
                    corrected_beta_low.x[:, np.newaxis]
                )

        return np.hstack([y_pred_low, y_pred_up])

    def __call__(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Call the ``transform`` method.

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : ArrayLike
            Target prediction

        z : ArrayLike
            Exogenous variable

        Returns
        -------
        NDArray
            features
        """
        return self.transform(X, y_pred, z)

    def __mul__(self, funct: Optional[Callable]) -> CCPCalibrator:
        """
        Multiply a ``CCPCalibrator`` with another function.
        This other function should return an array of shape (n_samples, 1)
        or (n_samples, ).

        The output of the ``transform`` method of the resulting
        ``CCPCalibrator`` instance will be multiplied by the ``funct`` values.

        Parameters
        ----------
        funct : Optional[Callable]
            function which should return an array of shape (n_samples, 1)
        or (n_samples, )

        Returns
        -------
        CCPCalibrator
            self, with ``funct`` append in the ``_multipliers`` argument list.
        """
        if funct is None:
            return self
        else:
            check_custom_calibrator_functions([funct])
            old_multipliers = self._multipliers
            new_calibrator = cast(CCPCalibrator, clone(self))
            if old_multipliers is None:
                new_calibrator._multipliers = [funct]
            else:
                new_calibrator._multipliers = old_multipliers + [funct]
            return new_calibrator

    def __rmul__(self, other) -> CCPCalibrator:
        """
        Do the same as ``__mul__``
        """
        return self.__mul__(other)
