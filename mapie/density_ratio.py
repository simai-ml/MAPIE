from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from mapie._typing import ArrayLike


class DensityRatioEstimator():
    """ Template class for density ratio estimation. """

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def check_is_fitted(self) -> None:
        raise NotImplementedError


class ProbClassificationDRE(DensityRatioEstimator):
    """
    Density ratio estimation by classification.

    This class implements the density ratio estimation by classification
    strategy. The broad idea is to first learn a discriminative classifier to
    distinguish between source and target datasets, and then use the class
    probability estimates from the classifier to estimate the density ratio.

    Parameters
    ----------
    estimator: Optional[ClassifierMixin]
        Any classifier with scikit-learn API (i.e. with fit, predict, and
        predict_proba methods), by default ``None``.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    clip_min: Optional[float]
        Lower bound the probability estimate from the classifier to
        ``clip_min``. If ``None``, the estimates are not lower bounded.

        By default ``None``.

    clip_max: Optional[float]
        Upper bound the probability estimate from the classifier to
        ``clip_max``. If ``None``, the estimates are not upper bounded.

        By default ``None``.

    Attributes
    ----------
    source_prob: float
        The marginal probability of getting a datapoint from the source
        distribution.

    target_prob: float
        The marginal probability of getting a datapoint from the target
        distribution.

    References
    ----------

    Examples
    --------

    """

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:

        self.estimator = self._check_estimator(estimator)

        if clip_max is None:
            self.clip_max = 1
        elif all((clip_max >= 0, clip_max <= 1)):
            self.clip_max = clip_max
        else:
            raise ValueError("Expected `clip_max` to be between 0 and 1.")

        if clip_min is None:
            self.clip_min = 0
        elif all((clip_min >= 0, clip_min <= clip_max)):
            self.clip_min = clip_min
        else:
            raise ValueError(
                "Expected `clip_min` to be between 0 and `clip_max`.")

    def _check_estimator(
        self,
        estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.

        Parameters
        ----------
        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.
        """
        if estimator is None:
            return LogisticRegression(class_weight="balanced", random_state=0)

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

        return estimator

    def fit(
        self,
        X_source: ArrayLike,
        X_target: ArrayLike,
        source_prob: Optional[float] = None,
        target_prob: Optional[float] = None,
        sample_weight: Optional[ArrayLike] = None
    ) -> ProbClassificationDRE:
        """
        Fit the discriminative classifier to source and target samples.

        Parameters
        ----------
        X_source: ArrayLike of shape (n_source_samples, n_features)
            Training data.

        X_target: ArrayLike of shape (n_target_samples, n_features)
            Training data.

        source_prob: Optional[float]
            The marginal probability of getting a datapoint from the source
            distribution. If ``None``, the proportion of source examples in
            the training dataset is used.

            By default ``None``.

        target_prob: Optional[float]
            The marginal probability of getting a datapoint from the target
            distribution. If ``None``, the proportion of target examples in
            the training dataset is used.

            By default ``None``.

        sample_weight : Optional[ArrayLike] of shape (n_source + n_target,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default ``None``.

        Returns
        -------
        ProbClassificationDRE
            The density ratio estimator itself.
        """

        # Find the marginal source and target probability.
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]

        if source_prob is None:
            source_prob = n_source / (n_source + n_target)

        if target_prob is None:
            target_prob = n_target / (n_source + n_target)

        if source_prob + target_prob != 1:
            raise ValueError(
                "``source_prob`` and ``target_prob`` do not add up to 1."
            )

        self.source_prob = source_prob
        self.target_prob = target_prob

        # Estimate the conditional probability of source/target given X.
        X = np.concatenate((X_source, X_target), axis=0)
        y = np.concatenate((np.zeros(n_source), np.ones(n_target)), axis=0)

        if type(self.estimator) == Pipeline:
            step_name = self.estimator.steps[-1][0]
            self.estimator.fit(
                X, y, **{f'{step_name}__sample_weight': sample_weight}
            )
        else:
            self.estimator.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> ArrayLike:
        """
        Predict the density ratio estimates for new samples.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Samples to get the density ratio estimates for.

        Returns
        -------
        ProbClassificationDRE
            The density ratio estimtor itself.
        """

        # Some models in sklearn have predict_proba but not predict_log_proba.
        if not hasattr(self.estimator, "predict_log_proba"):
            probs = self.estimator.predict_proba(X)
            log_probs = np.log(probs)
        else:
            log_probs = self.estimator.predict_log_proba(X)

        # Clip prob to mitigate extremely high or low dre.
        log_probs = np.clip(
            log_probs, a_min=np.log(self.clip_min), a_max=np.log(self.clip_max)
        )

        return np.exp(
            log_probs[:, 1]
            - log_probs[:, 0]
            + np.log(self.source_prob)
            - np.log(self.target_prob)
        )

    def check_is_fitted(self) -> None:
        if isinstance(self.estimator, Pipeline):
            check_is_fitted(self.estimator[-1])
        else:
            check_is_fitted(self.estimator)


def calculate_ess(weights: ArrayLike) -> float:
    """
    Calculates the effective sample size given importance weights for the
    source distribution.

    Parameters
    ----------
    weights: ArrayLike
        Importance weights for the examples in source distribution.
    """
    num = weights.sum()**2
    denom = (weights**2).sum()
    return num/denom
