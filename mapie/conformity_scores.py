from abc import ABCMeta, abstractmethod

import numpy as np

from .density_ratio import DensityRatioEstimator
from ._machine_precision import EPSILON
from ._typing import ArrayLike, NDArray


class ConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    sym : bool
        Whether to consider the conformity score as symmetrical or not.
    consistency_check : bool, optional
        Whether to check the consistency between the following methods:
        - get_estimation_distribution and
        - get_signed_conformity_scores
        by default True.
    eps : float, optional
        Threshold to consider when checking the consistency between the
        following methods:
        - get_estimation_distribution and
        - get_signed_conformity_scores
        The following equality must be verified:
        self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred)
        ) == y
        It should be specified if consistency_check==True.
        by default sys.float_info.epsilon.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        compute_score_weights: bool = False,
        eps: np.float64 = np.float64(1e-8),
    ):
        """
        Parameters
        ----------
        sym : bool
            Whether to consider the conformity score as symmetrical or not.
        consistency_check : bool, optional
            Whether to check the consistency between the following methods:
            - get_estimation_distribution and
            - get_signed_conformity_scores
            by default True.
        compute_score_weights : TODO
        eps : float, optional
            Threshold to consider when checking the consistency between the
            following methods:
            - get_estimation_distribution and
            - get_signed_conformity_scores
            The following equality must be verified:
            self.get_estimation_distribution(
                y_pred, self.get_conformity_scores(y, y_pred)
            ) == y
            It should be specified if consistency_check==True.
            by default sys.float_info.epsilon.
        """
        self.sym = sym
        self.consistency_check = consistency_check
        self.compute_score_weights = compute_score_weights
        self.eps = eps

    @abstractmethod
    def get_signed_conformity_scores(
        self, y: ArrayLike, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Placeholder for get_signed_conformity_scores.
        Subclasses should implement this method!

        Compute the signed conformity scores from the predicted values
        and the observed ones.

        Parameters
        ----------
        y : NDArray
            Observed values.
        y_pred : NDArray
            Predicted values.

        Returns
        -------
        NDArray
            Unsigned conformity scores.
        """

    @abstractmethod
    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Placeholder for get_estimation_distribution.
        Subclasses should implement this method!

        Compute samples of the estimation distribution from the predicted
        values and the conformity scores.

        Parameters
        ----------
        y_pred : NDArray
            Predicted values.
        conformity_scores : NDArray
            Conformity scores.

        Returns
        -------
        NDArray
            Observed values.
        """

    def check_consistency(
        self,
        y: NDArray,
        y_pred: NDArray,
        conformity_scores: NDArray,
    ) -> None:
        """
        Check consistency between the following methods:
        get_estimation_distribution and get_signed_conformity_scores

        The following equality should be verified:
        self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred)
        ) == y

        Parameters
        ----------
        y : NDArray
            Observed values.
        y_pred : NDArray
            Predicted values.
        conformity_scores: NDArray
            Conformity scores.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        score_distribution = self.get_estimation_distribution(y_pred, conformity_scores)
        abs_conformity_scores = np.abs(np.subtract(score_distribution, y))
        max_conf_score = np.max(abs_conformity_scores)
        if max_conf_score > self.eps:
            raise ValueError(
                "The two functions get_conformity_scores and "
                "get_estimation_distribution of the ConformityScore class"
                " are not consistent. "
                "The following equation must be verified: "
                "self.get_estimation_distribution(y_pred, "
                "self.get_conformity_scores(y, y_pred)) == y. "  # noqa: E501
                f"The maximum conformity score is {max_conf_score}."
                "The eps attribute may need to be increased if you are "
                "sure that the two methods are consistent."
            )

    def get_weights(self, X: NDArray) -> NDArray:
        """
        Compute weights for conformity scores on calibration samples.

        Parameters
        ----------
        X : NDArray
            Dataset used for computing the weights

        Returns
        -------
        NDArray
            Estimated weights
        """
        return np.ones(shape=(len(X) + 1))

    def get_conformity_scores(self, y: ArrayLike, y_pred: ArrayLike) -> NDArray:
        """
        Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        y : NDArray
            Observed values.
        y_pred : NDArray
            Predicted values.

        Returns
        -------
        NDArray
            Conformity scores.
        """
        conformity_scores = self.get_signed_conformity_scores(y, y_pred)
        if self.consistency_check:
            self.check_consistency(y, y_pred, conformity_scores)
        if self.sym:
            conformity_scores = np.abs(conformity_scores)
        return conformity_scores


class AbsoluteConformityScore(ConformityScore):
    """
    Absolute conformity score.

    The signed conformity score = y - y_pred.
    The conformity score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and
    its range is approximatively the same over the range of predicted values.
    """

    def __init__(self) -> None:
        super().__init__(sym=True, consistency_check=True)

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score
        """
        return np.add(y_pred, conformity_scores)


class GammaConformityScore(ConformityScore):
    """
    Gamma conformity score.

    The signed conformity score = (y - y_pred) / y_pred.
    The conformity score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and
    its range depends on the predicted values. Like the Gamma distribution,
    its support is limited to strictly positive reals.
    """

    def __init__(self) -> None:
        super().__init__(sym=False, consistency_check=False, eps=EPSILON)

    def _check_observed_data(self, y: ArrayLike) -> None:
        if not self._all_strictly_positive(y):
            raise ValueError(
                f"At least one of the observed target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    def _check_predicted_data(self, y_pred: ArrayLike) -> None:
        if not self._all_strictly_positive(y_pred):
            raise ValueError(
                f"At least one of the predicted target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    def _all_strictly_positive(self, y: ArrayLike) -> bool:
        if np.any(np.less_equal(y, 0)):
            return False
        return True

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        self._check_observed_data(y)
        self._check_predicted_data(y_pred)
        return np.divide(np.subtract(y, y_pred), y_pred)

    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)
        """
        self._check_predicted_data(y_pred)
        return np.multiply(y_pred, np.add(1, conformity_scores))


class CovariateShiftConformityScore(ConformityScore):
    def __init__(
        self,
        density_ratio_estimator: DensityRatioEstimator,
    ) -> None:
        super().__init__(
            sym=True,
            consistency_check=False,
            compute_score_weights=True,
        )
        self.density_ratio_estimator = density_ratio_estimator

    def get_signed_conformity_scores(self, y: NDArray, y_pred: NDArray) -> NDArray:
        scores = np.subtract(y, y_pred)
        return scores

    def get_weights(self, X: NDArray) -> NDArray:
        dre_test = self.density_ratio_estimator.predict(X)  # n_test
        dre_calib = self.density_ratio_estimator.calib_dr_estimates_  # n_calib
        denom = dre_calib.sum() + dre_test  # n_test
        calib_weights = dre_calib[:, np.newaxis] / denom  # (n_calib, n_test)
        test_weights = dre_test / denom  # n_test
        weights_stacked = np.vstack([calib_weights, test_weights]).T
        return weights_stacked  # (n_test, n_calib+1)

    def get_estimation_distribution(
        self, y_pred: NDArray, conformity_scores: NDArray
    ) -> NDArray:
        return np.add(y_pred, conformity_scores)
