from abc import ABCMeta, abstractmethod

import numpy as np

from mapie._typing import ArrayLike, NDArray


class ConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    sym: bool
        Whether to consider the conformity score as symmetrical or not.

    consistency_check: bool, optional
        Whether to check the consistency between the following methods:
        - ``get_estimation_distribution`` and
        - ``get_signed_conformity_scores``

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency between the
        following methods:
        - ``get_estimation_distribution`` and
        - ``get_signed_conformity_scores``
        The following equality must be verified:
        ``self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred)
        ) == y``
        It should be specified if ``consistency_check==True``.

        By default ``np.float64(1e-8)``.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: np.float64 = np.float64(1e-8),
    ):
        self.sym = sym
        self.consistency_check = consistency_check
        self.eps = eps

    @abstractmethod
    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Placeholder for ``get_signed_conformity_scores``.
        Subclasses should implement this method!

        Compute the signed conformity scores from the predicted values
        and the observed ones.

        Parameters
        ----------
        y: NDArray
            Observed values.

        y_pred: NDArray
            Predicted values.

        Returns
        -------
        NDArray
            Unsigned conformity scores.
        """

    @abstractmethod
    def get_estimation_distribution(
        self,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
    ) -> NDArray:
        """
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution from the predicted
        values and the conformity scores.

        Parameters
        ----------
        y_pred: NDArray
            Predicted values.

        conformity_scores: NDArray
            Conformity scores.

        Returns
        -------
        NDArray
            Observed values.
        """

    def check_consistency(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
    ) -> None:
        """
        Check consistency between the following methods:
        ``get_estimation_distribution`` and ``get_signed_conformity_scores``

        The following equality should be verified:
        ``self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred)
        ) == y``

        Parameters
        ----------
        y: NDArray
            Observed values.

        y_pred: NDArray
            Predicted values.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        score_distribution = self.get_estimation_distribution(
            y_pred, conformity_scores
        )
        abs_conformity_scores = np.abs(np.subtract(score_distribution, y))
        max_conf_score = np.max(abs_conformity_scores)
        if max_conf_score > self.eps:
            raise ValueError(
                "The two functions get_conformity_scores and "
                "get_estimation_distribution of the ConformityScore class "
                "are not consistent. "
                "The following equation must be verified: "
                "self.get_estimation_distribution(y_pred, "
                "self.get_conformity_scores(y, y_pred)) == y. "  # noqa: E501
                f"The maximum conformity score is {max_conf_score}."
                "The eps attribute may need to be increased if you are "
                "sure that the two methods are consistent."
            )

    def get_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        y: NDArray
            Observed values.

        y_pred: NDArray
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
