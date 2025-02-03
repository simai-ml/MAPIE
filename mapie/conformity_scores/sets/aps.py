from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.dummy import check_random_state
from sklearn.calibration import label_binarize

from mapie.conformity_scores.sets.naive import NaiveConformityScore
from mapie.conformity_scores.sets.utils import (
    check_include_last_label, check_proba_normalized
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.utils import compute_quantiles


class APSConformityScore(NaiveConformityScore):
    """
    Adaptive Prediction Sets (APS) method-based non-conformity score.
    It is based on the sum of the softmax outputs of the labels until the true
    label is reached, on the calibration set. See [1] for more details.

    References
    ----------
    [1] Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

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

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
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

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of predictions.
        """
        y_pred_proba = estimator.predict(X, agg_scores)
        # y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )
        return y_pred_proba

    @staticmethod
    def get_true_label_cumsum_proba(
        y: ArrayLike,
        y_pred_proba: NDArray,
        classes: ArrayLike
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the cumsumed probability of the true label.

        Parameters
        ----------
        y: NDArray of shape (n_samples, )
            Array with the labels.

        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Predictions of the model.

        classes: NDArray of shape (n_classes, )
            Array with the classes.

        Returns
        -------
        Tuple[NDArray, NDArray] of shapes (n_samples, 1) and (n_samples, ).
            The first element is the cumsum probability of the true label.
            The second is the sorted position of the true label.
        """
        y_true = label_binarize(y=y, classes=classes)
        index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
        y_pred_sorted = np.take_along_axis(y_pred_proba, index_sorted, axis=1)
        y_true_sorted = np.take_along_axis(y_true, index_sorted, axis=1)
        y_pred_sorted_cumsum = np.cumsum(y_pred_sorted, axis=1)
        cutoff = np.argmax(y_true_sorted, axis=1)
        true_label_cumsum_proba = np.take_along_axis(
            y_pred_sorted_cumsum, cutoff.reshape(-1, 1), axis=1
        )
        cutoff += 1

        return true_label_cumsum_proba, cutoff

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        y_enc: Optional[NDArray] = None,
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

        y_enc: NDArray of shape (n_samples,)
            Target values as normalized encodings.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        # Casting
        y_enc = cast(NDArray, y_enc)
        classes = cast(NDArray, self.classes)

        # Conformity scores
        conformity_scores, self.cutoff = (
            self.get_true_label_cumsum_proba(y, y_pred, classes)
        )
        y_proba_true = np.take_along_axis(
            y_pred, y_enc.reshape(-1, 1), axis=1
        )
        
        random_state = check_random_state(self.random_state)
        u = random_state.uniform(size=len(y_pred)).reshape(-1, 1)
        conformity_scores -= u * y_proba_true

        return conformity_scores

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
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

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = compute_quantiles(conformity_scores, alpha_np)
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_

    def _compute_v_parameter(
        self,
        y_proba_last_cumsumed: NDArray,
        threshold: NDArray,
        y_pred_proba_last: NDArray,
        prediction_sets: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Compute the V parameters from Romano+(2020).

        Parameters
        -----------
        y_proba_last_cumsumed: NDArray of shape (n_samples, n_alpha)
            Cumulated score of the last included label.

        threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
            Threshold to compare with y_proba_last_cumsum.

        y_pred_proba_last: NDArray of shape (n_samples, 1, n_alpha)
            Last included probability.

        predicition_sets: NDArray of shape (n_samples, n_alpha)
            Prediction sets.

        Returns
        --------
        NDArray of shape (n_samples, n_alpha)
            Vs parameters.
        """
        # compute V parameter from Romano+(2020)
        v_param = (
            (y_proba_last_cumsumed - threshold.reshape(1, -1)) /
            y_pred_proba_last[:, 0, :]
        )
        return v_param

    def _add_random_tie_breaking(
        self,
        prediction_sets: NDArray,
        y_pred_index_last: NDArray,
        y_pred_proba_cumsum: NDArray,
        y_pred_proba_last: NDArray,
        threshold: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Randomly remove last label from prediction set based on the
        comparison between a random number and the difference between
        cumulated score of the last included label and the quantile.

        Parameters
        ----------
        prediction_sets: NDArray of shape
            (n_samples, n_classes, n_threshold)
            Prediction set for each observation and each alpha.

        y_pred_index_last: NDArray of shape (n_samples, threshold)
            Index of the last included label.

        y_pred_proba_cumsum: NDArray of shape (n_samples, n_classes)
            Cumsumed probability of the model in the original order.

        y_pred_proba_last: NDArray of shape (n_samples, 1, threshold)
            Last included probability.

        threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
            Threshold to compare with y_proba_last_cumsum, can be either:

            - the quantiles associated with alpha values when
                ``cv`` == "prefit", ``cv`` == "split"
                or ``agg_scores`` is "mean"

            - the conformity score from training samples otherwise (i.e., when
            ``cv`` is CV splitter and ``agg_scores`` is "crossval")

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
            Updated version of prediction_sets with randomly removed labels.
        """
        # get cumsumed probabilities up to last retained label
        y_proba_last_cumsumed = np.squeeze(
            np.take_along_axis(
                y_pred_proba_cumsum,
                y_pred_index_last,
                axis=1
            ), axis=1
        )

        # get the V parameter from Romano+(2020) or Angelopoulos+(2020)
        v_param = self._compute_v_parameter(
            y_proba_last_cumsumed,
            threshold,
            y_pred_proba_last,
            prediction_sets
        )

        # get random numbers for each observation and alpha value
        random_state = check_random_state(self.random_state)
        random_state = cast(np.random.RandomState, random_state)
        u_param = random_state.uniform(size=(prediction_sets.shape[0], 1))
        # remove last label from comparison between uniform number and V
        label_to_keep = np.less_equal(v_param - u_param, EPSILON)
        np.put_along_axis(
            prediction_sets,
            y_pred_index_last,
            label_to_keep[:, np.newaxis, :],
            axis=1
        )
        return prediction_sets

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        include_last_label: Optional[Union[bool, str]] = True,
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

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in prediction sets.
            Choose among ``False``, ``True``  or ``"randomized"``.

            By default, ``True``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        include_last_label = check_include_last_label(include_last_label)

        # specify which thresholds will be used
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            thresholds = self.quantiles_
        else:
            thresholds = conformity_scores.ravel()

        # sort labels by decreasing probability
        y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last = (
            self._get_last_included_proba(
                y_pred_proba,
                thresholds,
                include_last_label,
                prediction_phase=True,
                **kwargs
            )
        )
        # get the prediction set by taking all probabilities above the last one
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            y_pred_included = np.greater_equal(
                y_pred_proba - y_pred_proba_last, -EPSILON
            )
        else:
            y_pred_included = np.less_equal(
                y_pred_proba - y_pred_proba_last, EPSILON
            )
        # remove last label randomly
        if include_last_label == "randomized":
            y_pred_included = self._add_random_tie_breaking(
                y_pred_included,
                y_pred_index_last,
                y_pred_proba_cumsum,
                y_pred_proba_last,
                thresholds,
                **kwargs
            )
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            prediction_sets = y_pred_included
        else:
            # compute the number of times the inequality is verified
            prediction_sets_summed = y_pred_included.sum(axis=2)
            prediction_sets = np.less_equal(
                prediction_sets_summed[:, :, np.newaxis]
                - self.quantiles_[np.newaxis, np.newaxis, :],
                EPSILON
            )

        return prediction_sets
