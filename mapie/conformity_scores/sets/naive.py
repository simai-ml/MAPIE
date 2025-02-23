from typing import Tuple, Union

import numpy as np

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import (
    check_proba_normalized, get_last_index_included
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import NDArray


class NaiveConformityScore(BaseClassificationScore):
    """
    Naive classification non-conformity score method that is based on the
    cumulative sum of probabilities until the 1-alpha threshold.

    Attributes
    ----------
    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Get the conformity score.

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
        conformity_scores = np.empty(y_pred.shape, dtype="float")
        return conformity_scores

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Get predictions from an EnsembleClassifier.

        Parameters
        -----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of predictions.
        """
        y_pred_proba = estimator.predict(X, agg_scores='mean')
        # y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        y_pred_proba = np.repeat(
            y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
        )
        return y_pred_proba

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Get the quantiles of the conformity scores for each uncertainty level.

        Parameters
        -----------
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        quantiles_ = 1 - alpha_np
        return quantiles_

    def _add_regularization(
        self,
        y_pred_proba_sorted_cumsum: NDArray,
        **kwargs
    ):
        """
        Add regularization to the sorted cumulative sum of predicted
        probabilities.

        Parameters
        ----------
        y_pred_proba_sorted_cumsum: NDArray of shape (n_samples, n_classes)
            The sorted cumulative sum of predicted probabilities.

        **kwargs: dict, optional
            Additional keyword arguments that might be used.
            The current implementation does not use any.

        Returns
        -------
        NDArray
            The adjusted cumulative sum of predicted probabilities after
            applying the regularization technique.
        """
        return y_pred_proba_sorted_cumsum

    def _get_last_included_proba(
        self,
        y_pred_proba: NDArray,
        thresholds: NDArray,
        include_last_label: Union[bool, str, None],
        **kwargs
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Function that returns the smallest score
        among those which are included in the prediction set.

        Parameters
        ----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Predictions of the model.

        thresholds: NDArray of shape (n_alphas, )
            Quantiles that have been computed from the conformity scores.

        include_last_label: Union[bool, str, None]
            Whether to include or not the label whose score exceeds threshold.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            Arrays of shape (n_samples, n_classes, n_alphas),
            (n_samples, 1, n_alphas) and (n_samples, 1, n_alphas).
            They are respectively the cumsumed scores in the original
            order which can be different according to the value of alpha
            with the RAPS method, the index of the last included score
            and the value of the last included score.
        """
        index_sorted = np.flip(
            np.argsort(y_pred_proba, axis=1), axis=1
        )
        # sort probabilities by decreasing order
        y_pred_proba_sorted = np.take_along_axis(
            y_pred_proba, index_sorted, axis=1
        )
        # get sorted cumulated score
        y_pred_proba_sorted_cumsum = np.cumsum(y_pred_proba_sorted, axis=1)
        y_pred_proba_sorted_cumsum = self._add_regularization(
            y_pred_proba_sorted_cumsum, **kwargs
        )  # Do nothing as no regularization for the naive method

        # get cumulated score at their original position
        y_pred_proba_cumsum = np.take_along_axis(
            y_pred_proba_sorted_cumsum,
            np.argsort(index_sorted, axis=1),
            axis=1
        )
        # get index of the last included label
        y_pred_index_last = get_last_index_included(
            y_pred_proba_cumsum,
            thresholds,
            include_last_label
        )
        # get the probability of the last included label
        y_pred_proba_last = np.take_along_axis(
            y_pred_proba,
            y_pred_index_last,
            axis=1
        )

        zeros_scores_proba_last = (y_pred_proba_last <= EPSILON)

        # If the last included proba is zero, change it to the
        # smallest non-zero value to avoid inluding them in the
        # prediction sets.
        if np.sum(zeros_scores_proba_last) > 0:
            y_pred_proba_last[zeros_scores_proba_last] = np.expand_dims(
                np.min(
                    np.ma.masked_less(
                        y_pred_proba,
                        EPSILON
                    ).filled(fill_value=np.inf),
                    axis=1
                ), axis=1
            )[zeros_scores_proba_last]

        return y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Generate prediction sets based on the probability predictions,
        the conformity scores and the uncertainty level.

        Parameters
        -----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        # sort labels by decreasing probability
        _, _, y_pred_proba_last = (
            self._get_last_included_proba(
                y_pred_proba,
                thresholds=self.quantiles_,
                include_last_label=True
            )
        )
        # get the prediction set by taking all probabilities above the last one
        prediction_sets = np.greater_equal(
            y_pred_proba - y_pred_proba_last, -EPSILON
        )

        return prediction_sets
