from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted

from mapie._compatibility import np_nanquantile
from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all
from .regression import MapieRegressor
from mapie.utils import check_alpha, check_alpha_and_n_samples


class MapieTimeSeriesRegressor(MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.

    This class implements the EnbPI strategy for estimating
    prediction intervals on single-output time series. The only valid
    ``method`` is ``"enbpi"``.

    Actually, EnbPI only corresponds to ``MapieTimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap``.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107
    """

    cv_need_agg_function_ = MapieRegressor.cv_need_agg_function_ \
        + ["BlockBootstrap"]
    valid_methods_ = ["enbpi"]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            random_state=random_state
        )

    def _relative_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """
        Compute the conformity scores on a data set.

        Parameters
        ----------
            X : ArrayLike of shape (n_samples, n_features)
            Input data.

            y : ArrayLike of shape (n_samples,)
                Input labels.

        Returns
        -------
            The conformity scores corresponding to the input data set.
        """
        y_pred, _ = super().predict(X, alpha=0.5, ensemble=True)
        return np.asarray(y) - np.asarray(y_pred)

    def _beta_optimize(
        self,
        alpha: Union[float, NDArray],
        upper_bounds: NDArray,
        lower_bounds: NDArray,
    ) -> NDArray:
        """
        Minimize the width of the PIs, for a given difference of quantiles.

        Parameters
        ----------
        alpha: Union[float, NDArray]
            The quantiles to compute.
        upper_bounds: NDArray
            The array of upper values.
        lower_bounds: NDArray
            The array of lower values.

        Returns
        -------
        NDArray
            Array of betas minimizing the differences
            ``(1-alpa+beta)-quantile - beta-quantile``.

        Raises
        ------
        ValueError
            If lower and upper bounds arrays don't have the same shape.
        """
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                "Lower and upper bounds arrays should have the same shape."
            )
        alpha = cast(NDArray, alpha)
        betas_0 = np.full(
            shape=(len(lower_bounds), len(alpha)),
            fill_value=np.nan,
            dtype=float,
        )

        for ind_alpha, _alpha in enumerate(alpha):
            betas = np.linspace(
                _alpha / (len(lower_bounds) + 1),
                _alpha,
                num=len(lower_bounds),
                endpoint=True,
            )
            one_alpha_beta = np_nanquantile(
                upper_bounds.astype(float),
                1 - _alpha + betas,
                axis=1,
                method="higher",
            )
            beta = np_nanquantile(
                lower_bounds.astype(float),
                betas,
                axis=1,
                method="lower",
            )
            betas_0[:, ind_alpha] = betas[
                np.argmin(one_alpha_beta - beta, axis=0)
            ]

        return betas_0

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        """
        Compared to the method ``fit`` of ``MapieRegressor``, the ``fit``
        method of ``MapieTimeSeriesRegressor`` computes the
        ``conformity_scores_`` with relative values.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no conformity scores.
            If weights are non-uniform,
            conformity scores are still uniformly weighted.

            By default ``None``.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        self = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.conformity_scores_ = self._relative_conformity_scores(X, y)
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> MapieTimeSeriesRegressor:
        """
        Update the ``conformity_scores_`` attribute when new data with known
        labels are available.
        Note: Don't use ``partial_fit`` with samples of the training set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of ``y`` is greater than
            the length of the training set.
        """
        check_is_fitted(self, self.fit_attributes)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        n = len(self.conformity_scores_)
        if len(X) > n:
            raise ValueError(
                "The number of observations to update is higher than the"
                "number of training instances."
            )
        new_conformity_scores_ = self._relative_conformity_scores(X, y)
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[
            -len(new_conformity_scores_):
        ] = new_conformity_scores_
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = True,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Correspond to 'Conformal prediction for dynamic time-series'.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
                - [:, 0, :]: Lower bound of the prediction interval.
                - [:, 1, :]: Upper bound of the prediction interval.
        """
        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        y_pred = self.single_estimator_.predict(X)
        n = len(self.conformity_scores_)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)
        check_alpha_and_n_samples(alpha_np, n)

        if optimize_beta:
            betas_0 = self._beta_optimize(
                alpha_np,
                self.conformity_scores_.reshape(1, -1),
                self.conformity_scores_.reshape(1, -1),
            )
        else:
            betas_0 = np.repeat(alpha[:, np.newaxis] / 2, n, axis=0)

        lower_quantiles = np_nanquantile(
            self.conformity_scores_.astype(float),
            betas_0[0, :],
            axis=0,
            method="lower",
        ).T
        higher_quantiles = np_nanquantile(
            self.conformity_scores_.astype(float),
            1 - alpha_np + betas_0[0, :],
            axis=0,
            method="higher",
        ).T
        self.lower_quantiles_ = lower_quantiles
        self.higher_quantiles_ = higher_quantiles

        if self.method in self.no_agg_methods_ \
                or self.cv in self.no_agg_cv_:
            y_pred_low = y_pred[:, np.newaxis] + lower_quantiles
            y_pred_up = y_pred[:, np.newaxis] + higher_quantiles
        else:
            y_pred_multi = self._pred_multi(X)
            pred = aggregate_all(self.agg_function, y_pred_multi)
            lower_bounds, upper_bounds = pred, pred

            y_pred_low = lower_bounds.reshape(-1, 1) + lower_quantiles
            y_pred_up = upper_bounds.reshape(-1, 1) + higher_quantiles

            if ensemble:
                y_pred = aggregate_all(self.agg_function, y_pred_multi)

        return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

    def _more_tags(self):
        return {
            "_xfail_checks":
            {
                "check_estimators_partial_fit_n_features":
                "partial_fit can only be called on fitted models"
            }
        }
