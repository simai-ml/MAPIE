from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, check_is_fitted, indexable)

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.sets.raps import RAPSConformityScore
from mapie.conformity_scores.utils import (
    check_depreciated_size_raps, check_classification_conformity_score,
    check_target
)
from mapie.estimator.classifier import EnsembleClassifier
from mapie.utils import (check_alpha, check_alpha_and_n_samples, check_cv,
                         check_estimator_classification, check_n_features_in,
                         check_n_jobs, check_null_weight, check_predict_params,
                         check_verbose)


class MapieClassifier(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for classification.

    This class implements several conformal prediction strategies for
    estimating prediction sets for classification. Instead of giving a
    single predicted label, the idea is to give a set of predicted labels
    (or prediction sets) which come with mathematically guaranteed coverages.

    Parameters
    ----------
    estimator: Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: Optional[str]
        [DEPRECIATED see instead conformity_score]
        Method to choose for prediction interval estimates.
        Choose among:

        - ``"naive"``, sum of the probabilities until the 1-alpha threshold.

        - ``"lac"`` (formerly called ``"score"``), Least Ambiguous set-valued
          Classifier. It is based on the scores
          (i.e. 1 minus the softmax score of the true label)
          on the calibration set. See [1] for more details.

        - ``"aps"`` (formerly called "cumulated_score"), Adaptive Prediction
          Sets method. It is based on the sum of the softmax outputs of the
          labels until the true label is reached, on the calibration set.
          See [2] for more details.

        - ``"raps"``, Regularized Adaptive Prediction Sets method. It uses the
          same technique as ``"aps"`` method but with a penalty term
          to reduce the size of prediction sets. See [3] for more
          details. For now, this method only works with ``"prefit"`` and
          ``"split"`` strategies.

        - ``"top_k"``, based on the sorted index of the probability of the true
          label in the softmax outputs, on the calibration set. In case two
          probabilities are equal, both are taken, thus, the size of some
          prediction sets may be different from the others. See [3] for
          more details.

        - ``None``, that does not specify the method used.

        In any case, the `method` parameter does not take precedence over the
        `conformity_score` parameter to define the method used.

        By default ``None``.

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation)
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, it will be set to 0.1.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    conformity_score: BaseClassificationScore
        Score function that handle all that is related to conformity scores.

        In any case, the `conformity_score` parameter takes precedence over the
        `method` parameter to define the method used.

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    verbose: int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    estimator_: EnsembleClassifier
        Sklearn estimator that handle all that is related to the estimator.

    conformity_score_function_: BaseClassificationScore
        Score function that handle all that is related to conformity scores.

    n_features_in_: int
        Number of features passed to the fit method.

    conformity_scores_: ArrayLike of shape (n_samples_train)
        The conformity scores used to calibrate the prediction sets.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``conformity_scores_`` and alpha values.

    label_encoder_: LabelEncoder
        Label encoder used to encode the labels.

    References
    ----------
    [1] Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    Journal of the American Statistical Association, 114, 2019.

    [2] Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

    [3] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import MapieClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> clf = GaussianNB().fit(X_toy, y_toy)
    >>> mapie = MapieClassifier(estimator=clf, cv="prefit").fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.2)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False False]
     [ True False False]
     [ True  True False]
     [ True  True False]
     [False  True False]
     [False  True  True]
     [False  True  True]
     [False False  True]
     [False False  True]]
    """

    fit_attributes = [
        "estimator_",
        "n_features_in_",
        "conformity_scores_",
        "conformity_score_function_",
        "classes_",
        "label_encoder_"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: Optional[str] = None,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        conformity_score: Optional[BaseClassificationScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.conformity_score = conformity_score
        self.random_state = random_state
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _get_classes_info(
            self, estimator: ClassifierMixin, y: NDArray
    ) -> Tuple[int, NDArray]:
        """
        Compute the number of classes and the classes values
        according to either the pre-trained model or to the
        values in y.

        Parameters
        ----------
        estimator: ClassifierMixin
            Estimator pre-fitted or not.

        y: NDArray
            Values to predict.

        Returns
        -------
        Tuple[int, NDArray]
            The number of unique classes and their unique
            values.

        Raises
        ------
        ValueError
            If `cv="prefit"` and that classes in `y` are not included into
            `estimator.classes_`.

        Warning
            If number of calibration labels is lower than number of labels
            for training (in prefit setting)
        """
        n_unique_y_labels = len(np.unique(y))
        if self.cv == "prefit":
            classes = estimator.classes_
            n_classes = len(np.unique(classes))
            if not set(np.unique(y)).issubset(classes):
                raise ValueError(
                    "Values in y do not matched values in estimator.classes_."
                    + " Check that you are not adding any new label"
                )
            if n_classes > n_unique_y_labels:
                warnings.warn(
                    "WARNING: your conformity dataset has less labels"
                    + " than your training dataset (training"
                    + f" has {n_classes} unique labels while"
                    + f" conformity have {n_unique_y_labels} unique labels"
                )

        else:
            n_classes = n_unique_y_labels
            classes = np.unique(y)

        return n_classes, classes

    def _get_label_encoder(self) -> LabelEncoder:
        """
        Construct the label encoder with respect to the classes values.

        Returns
        -------
        LabelEncoder
        """
        return LabelEncoder().fit(self.classes_)

    def _check_fit_parameter(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        size_raps: Optional[float] = None,
    ):
        """
        Perform several checks on class parameters.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Non-null sample weights.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default ``None``.

        Returns
        -------
        Tuple[Optional[ClassifierMixin],
        Optional[Union[int, str, BaseCrossValidator]],
        ArrayLike, NDArray, NDArray, Optional[NDArray],
        Optional[NDArray], ArrayLike]
            Parameters checked

        Raises
        ------
        ValueError
            If conformity score is FittedResidualNormalizing score and method
            is neither ``"prefit"`` or ``"split"``.

        ValueError
            If ``cv`` is `"prefit"`` or ``"split"`` and ``method`` is not
            ``"base"``.
        """
        self._check_parameters()
        cv = check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        X, y = indexable(X, y)
        y = _check_y(y)

        sample_weight = cast(Optional[NDArray], sample_weight)
        groups = cast(Optional[NDArray], groups)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        y = cast(NDArray, y)

        estimator = check_estimator_classification(X, y, cv, self.estimator)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)

        self.n_classes_, self.classes_ = self._get_classes_info(estimator, y)
        self.label_encoder_ = self._get_label_encoder()
        y_enc = self.label_encoder_.transform(y)

        cs_estimator = check_classification_conformity_score(
            conformity_score=self.conformity_score,
            method=self.method,
        )
        check_depreciated_size_raps(size_raps)
        cs_estimator.set_external_attributes(
            classes=self.classes_,
            label_encoder=self.label_encoder_,
            size_raps=size_raps,
            random_state=self.random_state
        )
        if (
            isinstance(cs_estimator, RAPSConformityScore) and
            not (
                self.cv in ["split", "prefit"] or
                isinstance(self.cv, BaseShuffleSplit)
            )
        ):
            raise ValueError(
                "RAPS method can only be used "
                "with ``cv='split'`` and ``cv='prefit'``."
            )

        # Cast
        X, y_enc, y = cast(NDArray, X), cast(NDArray, y_enc), cast(NDArray, y)
        sample_weight = cast(NDArray, sample_weight)
        groups = cast(NDArray, groups)

        X, y, y_enc, sample_weight, groups = \
            cs_estimator.split_data(X, y, y_enc, sample_weight, groups)
        self.n_samples_ = cs_estimator.n_samples_

        check_target(cs_estimator, y)

        return (
            estimator, cs_estimator, cv, X, y, y_enc, sample_weight, groups
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        size_raps: Optional[float] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any
    ) -> MapieClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: NDArray of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default ``None``.

        size_raps: Optional[float]
            Percentage of the data to be used for choosing lambda_star and
            k_star for the RAPS method.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        kwargs : dict
            Additional fit and predict parameters.

        Returns
        -------
        MapieClassifier
            The model itself.
        """
        fit_params = kwargs.pop('fit_params', {})
        predict_params = kwargs.pop('predict_params', {})

        if len(predict_params) > 0:
            self._predict_params = True
        else:
            self._predict_params = False

        # Checks
        (estimator,
         self.conformity_score_function_,
         cv,
         X,
         y,
         y_enc,
         sample_weight,
         groups) = self._check_fit_parameter(
            X, y, sample_weight, groups, size_raps
        )

        # Cast
        X, y_enc, y = cast(NDArray, X), cast(NDArray, y_enc), cast(NDArray, y)
        sample_weight = cast(NDArray, sample_weight)
        groups = cast(NDArray, groups)

        # Work
        self.estimator_ = EnsembleClassifier(
            estimator,
            self.n_classes_,
            cv,
            self.n_jobs,
            self.random_state,
            self.test_size,
            self.verbose,
        )
        # Fit the prediction function
        self.estimator_ = self.estimator_.fit(
            X, y, y_enc=y_enc, sample_weight=sample_weight, groups=groups,
            **fit_params
        )

        # Predict on calibration data
        y_pred_proba, y, y_enc = self.estimator_.predict_proba_calib(
            X, y, y_enc, groups, **predict_params
        )

        # Compute the conformity scores
        self.conformity_score_function_.set_ref_predictor(self.estimator_)
        self.conformity_scores_ = \
            self.conformity_score_function_.get_conformity_scores(
                y, y_pred_proba, y_enc=y_enc, X=X,
                sample_weight=sample_weight, groups=groups
            )
        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        include_last_label: Optional[Union[bool, str]] = True,
        agg_scores: Optional[str] = "mean",
        **predict_params
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction and prediction sets on new samples based on target
        confidence interval.
        Prediction sets for a given ``alpha`` are deduced from:

        - quantiles of softmax scores (``"lac"`` method)
        - quantiles of cumulated scores (``"aps"`` method)

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction sets.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in
            prediction sets for the "aps" method. Choose among:

            - False, does not include label whose cumulated score is just over
              the quantile.
            - True, includes label whose cumulated score is just over the
              quantile, unless there is only one label in the prediction set.
            - "randomized", randomly includes label whose cumulated score is
              just over the quantile based on the comparison of a uniform
              number and the difference between the cumulated score of
              the last label and the quantile.

            When set to ``True`` or ``False``, it may result in a coverage
            higher than ``1 - alpha`` (because contrary to the "randomized"
            setting, none of these methods create empty prediction sets). See
            [2] and [3] for more details.

            By default ``True``.

        agg_scores: Optional[str]

            How to aggregate the scores output by the estimators on test data
            if a cross-validation strategy is used. Choose among:

            - "mean", take the mean of scores.
            - "crossval", compare the scores between all training data and each
              test point for each label to estimate if the label must be
              included in the prediction set. Follows algorithm 2 of
              Romano+2020.

            By default "mean".

        predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is None.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples,) and (n_samples, n_classes, n_alpha) if alpha is not None.
        """
        # Checks

        if hasattr(self, '_predict_params'):
            check_predict_params(self._predict_params,
                                 predict_params, self.cv)

        check_is_fitted(self, self.fit_attributes)
        alpha = cast(Optional[NDArray], check_alpha(alpha))

        # Estimate predictions
        y_pred = self.estimator_.single_estimator_.predict(X, **predict_params)
        if alpha is None:
            return y_pred

        # Estimate of probabilities from estimator(s)
        # In all cases: len(y_pred_proba.shape) == 3
        # with  (n_test, n_classes, n_alpha or n_train_samples)
        n = len(self.conformity_scores_)
        alpha_np = cast(NDArray, alpha)
        check_alpha_and_n_samples(alpha_np, n)

        # Estimate prediction sets
        prediction_sets = self.conformity_score_function_.predict_set(
            X, alpha_np,
            estimator=self.estimator_,
            conformity_scores=self.conformity_scores_,
            include_last_label=include_last_label,
            agg_scores=agg_scores,
        )

        self.quantiles_ = self.conformity_score_function_.quantiles_

        return y_pred, prediction_sets
