import logging
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from mapie.conformity_scores.interface import BaseConformityScore
from mapie.estimator.regressor import EnsembleRegressor

from mapie._compatibility import np_nanquantile
from mapie._machine_precision import EPSILON
from mapie._typing import NDArray


class BaseRegressionScore(BaseConformityScore, metaclass=ABCMeta):
    """
    Base conformity score class for regression task.

    This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    sym: bool
        Whether to consider the conformity score as symmetrical or not.

    consistency_check: bool, optional
        Whether to check the consistency between the methods
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        If ``True``, the following equality must be verified::

            y == self.get_estimation_distribution(
                y_pred,
                self.get_conformity_scores(y, y_pred, **kwargs),
                **kwargs)

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency
        between ``get_estimation_distribution`` and ``get_conformity_scores``.
        It should be specified if ``consistency_check==True``.

        By default, it is defined by the default precision.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ):
        super().__init__()
        self.sym = sym
        self.consistency_check = consistency_check
        self.eps = eps

    @abstractmethod
    def get_signed_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_conformity_scores``.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Signed conformity scores.
        """

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        conformity_scores = \
            self.get_signed_conformity_scores(y, y_pred, **kwargs)
        if self.consistency_check:
            self.check_consistency(y, y_pred, conformity_scores, **kwargs)
        if self.sym:
            conformity_scores = np.abs(conformity_scores)
        return conformity_scores

    def check_consistency(
        self,
        y: NDArray,
        y_pred: NDArray,
        conformity_scores: NDArray,
        **kwargs
    ) -> None:
        """
        Check consistency between the following methods:
        ``get_estimation_distribution`` and ``get_signed_conformity_scores``

        The following equality should be verified::

            y == self.get_estimation_distribution(
                y_pred,
                self.get_conformity_scores(y, y_pred, **kwargs),
                **kwargs)

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        score_distribution = self.get_estimation_distribution(
            y_pred, conformity_scores, **kwargs
        )
        abs_conformity_scores = np.abs(np.subtract(score_distribution, y))
        max_conf_score = np.max(abs_conformity_scores)
        if max_conf_score > self.eps:
            raise ValueError(
                "The two functions get_conformity_scores and "
                "get_estimation_distribution of the BaseRegressionScore class "
                "are not consistent. "
                "The following equation must be verified: "
                "self.get_estimation_distribution(y_pred, "
                "self.get_conformity_scores(y, y_pred)) == y. "
                f"The maximum conformity score is {max_conf_score}. "
                "The eps attribute may need to be increased if you are "
                "sure that the two methods are consistent."
            )

    @abstractmethod
    def get_estimation_distribution(
        self,
        y_pred: NDArray,
        conformity_scores: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution given the predicted
        targets and the conformity scores.

        Parameters
        ----------
        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        Returns
        -------
        NDArray of shape (n_samples,)
            Observed values.
        """

    @staticmethod
    def _beta_optimize(
        alpha_np: NDArray,
        upper_bounds: NDArray,
        lower_bounds: NDArray,
    ) -> NDArray:
        """
        Minimize the width of the PIs, for a given difference of quantiles.

        Parameters
        ----------
        alpha_np: NDArray
            The quantiles to compute.

        upper_bounds: NDArray of shape (n_samples,)
            The array of upper values.

        lower_bounds: NDArray of shape (n_samples,)
            The array of lower values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Array of betas minimizing the differences
            ``(1-alpha+beta)-quantile - beta-quantile``.
        """
        # Using logging.warning instead of warnings.warn to avoid warnings during tests
        logging.warning(
            "The option to optimize beta (minimize interval width) is not working and "
            "needs to be fixed. See more details in "
            "https://github.com/scikit-learn-contrib/MAPIE/issues/588"
        )

        beta_np = np.full(
            shape=(len(lower_bounds), len(alpha_np)),
            fill_value=np.nan,
            dtype=float,
        )

        for ind_alpha, _alpha in enumerate(alpha_np):
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
            beta_np[:, ind_alpha] = betas[
                np.argmin(one_alpha_beta - beta, axis=0)
            ]

        return beta_np

    def get_bounds(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleRegressor,
        conformity_scores: NDArray,
        ensemble: bool = False,
        method: str = 'base',
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Compute bounds of the prediction intervals from the observed values,
        the estimator of type ``EnsembleRegressor`` and the conformity scores.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        estimator: EnsembleRegressor
            Estimator that is fitted to predict y from X.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.

            By default ``False``.

        method: str
            Method to choose for prediction interval estimates.
            The ``"plus"`` method implies that the quantile is calculated
            after estimating the bounds, whereas the other methods
            (among the ``"naive"``, ``"base"`` or ``"minmax"`` methods,
            for example) do the opposite.

            By default ``base``.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

            By default ``False``.

        allow_infinite_bounds: bool
            Allow infinite prediction intervals to be produced.

            By default ``False``.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            - The predictions itself. (y_pred) of shape (n_samples,).
            - The lower bounds of the prediction intervals of shape
              (n_samples, n_alpha).
            - The upper bounds of the prediction intervals of shape
              (n_samples, n_alpha).

        Raises
        ------
        ValueError
            If beta optimisation with symmetrical conformity score function.
        """
        if self.sym and optimize_beta:
            raise ValueError(
                "Interval width minimization cannot be used with a "
                + "symmetrical conformity score function."
            )

        y_pred, y_pred_low, y_pred_up = estimator.predict(X, ensemble)
        signed = -1 if self.sym else 1

        if optimize_beta:
            beta_np = self._beta_optimize(
                alpha_np,
                conformity_scores.reshape(1, -1),
                conformity_scores.reshape(1, -1),
            )
        else:
            beta_np = alpha_np / 2

        if method == "plus":
            alpha_low = alpha_np if self.sym else beta_np
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np + beta_np

            conformity_scores_low = self.get_estimation_distribution(
                y_pred_low, signed * conformity_scores, X=X
            )
            conformity_scores_up = self.get_estimation_distribution(
                y_pred_up, conformity_scores, X=X
            )
            bound_low = self.get_quantile(
                conformity_scores_low, alpha_low, axis=1, reversed=True,
                unbounded=allow_infinite_bounds
            )
            bound_up = self.get_quantile(
                conformity_scores_up, alpha_up, axis=1,
                unbounded=allow_infinite_bounds
            )

        else:
            if self.sym:
                alpha_ref = 1-alpha_np
                quantile_ref = self.get_quantile(
                    conformity_scores[..., np.newaxis], alpha_ref, axis=0
                )
                quantile_low, quantile_up = -quantile_ref, quantile_ref

            else:
                alpha_low, alpha_up = beta_np, 1 - alpha_np + beta_np

                quantile_low = self.get_quantile(
                    conformity_scores[..., np.newaxis],
                    alpha_low, axis=0, reversed=True,
                    unbounded=allow_infinite_bounds
                )
                quantile_up = self.get_quantile(
                    conformity_scores[..., np.newaxis],
                    alpha_up, axis=0,
                    unbounded=allow_infinite_bounds
                )

            bound_low = self.get_estimation_distribution(
                y_pred_low, quantile_low, X=X
            )
            bound_up = self.get_estimation_distribution(
                y_pred_up, quantile_up, X=X
            )

        return y_pred, bound_low, bound_up

    def predict_set(
        self,
        X: NDArray,
        alpha_np: NDArray,
        **kwargs
    ):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence set.

        Parameters
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns
        --------
        The output structure depend on the ``get_bounds`` method.
            The prediction sets for each sample and each alpha level.
        """
        return self.get_bounds(X=X, alpha_np=alpha_np, **kwargs)
