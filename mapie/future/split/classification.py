from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.future.calibrators.utils import check_calibrator
from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.interface import BaseConformityScore
from mapie.conformity_scores.utils import check_classification_conformity_score
from mapie.estimator.classifier import EnsembleClassifier
from mapie.future.split.base import BaseCalibrator, SplitCP


class SplitCPClassifier(SplitCP):
    """
    Class to compute Conformal Predictions in a ``"split"`` approach for
    classification tasks.
    It is based on a predictor (a sklearn estimator), and a calibrator
    (``Calibrator`` object).

    Parameters
    ----------
    predictor: Optional[ClassifierMixin]
        Any classifier from scikit-learn API.
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, ``predictor`` defaults to a ``LogisticRegression``
        instance.

        By default ``"None"``.

    calibrator: Optional[BaseCalibrator]
        A ``BaseCalibrator`` instance used to estimate the conformity scores.

        If ``None``, use as default a ``StandardCalibrator`` instance.

        By default ``None``.

    cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]]
        The splitting strategy for computing conformity scores.
        Choose among:

        - Any splitter (``ShuffleSplit`` or ``PredefinedSplit``)
        with ``n_splits=1``.
        - ``"prefit"``, assumes that ``predictor`` has been fitted already.
          All data provided in the ``calibrate`` method is then used
          for the calibration.
          The user has to take care manually that data used for model fitting
          and calibration (the data given in the ``calibrate`` method)
          are disjoint.
        - ``"split"`` or ``None``: divide the data into training and
          calibration subsets (using the default ``calib_size=0.3``).
          The splitter used is the following:
            ``sklearn.model_selection.ShuffleSplit`` with ``n_splits=1``.

        By default ``None``.

    conformity_score: Optional[BaseClassificationScore]
        ``BaseClassificationScore`` instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteBaseClassificationScore``
        symetrical conformity score
        - Any ``BaseClassificationScore`` class

        By default ``None``.

    alpha: Optional[float]
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``None``

    random_state: Optional[int]
        Integer used to set the numpy seed, to get reproducible calibration
        results.
        If ``None``, the prediction intervals will be stochastic, and will
        change if you refit the calibration (even if no arguments have change).

        WARNING: If ``random_state``is not ``None``, ``np.random.seed`` will
        be changed, which will reset the seed for all the other random
        number generators. It may have an impact on the rest of your code.

        By default ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.future.split import SplitCPClassifier
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400,2).reshape(-1, 1)
    >>> y_train = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    >>> mapie_reg = SplitCPClassifier(alpha=0.1, random_state=1)
    >>> mapie_reg = mapie_reg.fit(X_train, y_train)
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    """
    def __init__(
        self,
        predictor: Optional[
            Union[
                ClassifierMixin,
                Pipeline,
                List[Union[ClassifierMixin, Pipeline]]
            ]
        ] = None,
        calibrator: Optional[BaseCalibrator] = None,
        cv: Optional[Union[str, PredefinedSplit, ShuffleSplit]] = None,
        alpha: Optional[float] = None,
        conformity_score: Optional[BaseClassificationScore] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.predictor = predictor
        self.conformity_score = conformity_score
        self.calibrator = calibrator
        self.alpha = alpha

    def _check_estimator_fit_predict_predict_proba(
        self, estimator: ClassifierMixin
    ) -> None:
        """
        Check that the estimator has a fit and precict method.

        Parameters
        ----------
        estimator: ClassifierMixin
            Estimator to train.

        Raises
        ------
        ValueError
            If the estimator does not have a fit or predict or predict_proba
            attribute.
        """
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")
                and hasattr(estimator, "predict_proba")):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )

    def _check_estimator_classification(
        self,
        estimator: Optional[ClassifierMixin] = None,
        cv: Optional[Union[str, PredefinedSplit, ShuffleSplit]] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[ClassifierMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` nor ``predict_proba`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``cv`` attribute is ``"prefit"``.
        """
        if estimator is None:
            estimator = LogisticRegression(multi_class="multinomial")

        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        self._check_estimator_fit_predict_predict_proba(est)

        if cv == "prefit":
            check_is_fitted(est)
            if not hasattr(est, "classes_"):
                raise AttributeError(
                    "Fitted classifier must contain 'classes_' attribute."
                )
        return est

    def _check_fit_parameters(self) -> ClassifierMixin:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        Copy the ``predictor`` in ``predictor_`` attribute if ``cv="prefit"``.
        """
        self.cv = self._check_cv(self.cv)
        predictor = self._check_estimator_classification(self.predictor,
                                                         self.cv)
        return predictor

    def _check_calibrate_parameters(self) -> Tuple[
        BaseClassificationScore, BaseCalibrator
    ]:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """
        conformity_score_ = check_classification_conformity_score(
            self.conformity_score, None
        )
        calibrator = check_calibrator(self.calibrator)
        calibrator.sym = True
        calibrator.alpha = self.alpha
        calibrator.random_state = self.random_state
        self._check_alpha(self.alpha)
        return conformity_score_, calibrator

    def get_conformity_scores(
        self,
        conformity_score: BaseConformityScore,
        X: NDArray,
        y: NDArray,
        y_pred: NDArray,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs,
    ) -> NDArray:
        """
        Return the conformity scores of the data

        Parameters
        ----------
        conformity_score: BaseRegressionScore
            Score function that handle all that is related
            to conformity scores.

        X: NDArray of shape (n_samples, n_features)
            Data

        y: NDArray of shape (n_samples,)
            Target

        y_pred: NDArray of shape (n_samples,)
            Predictions

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights of the data, used as weights in the
            calibration process.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        y_enc = LabelEncoder().fit(self.predictor_.classes_).transform(y)
        conformity_score = cast(BaseClassificationScore, conformity_score)

        conformity_score.set_external_attributes(
            classes=self.predictor_.classes_,
            random_state=self.random_state,
        )

        return conformity_score.get_conformity_scores(
            y, y_pred, y_enc=y_enc, X=X,
            sample_weight=sample_weight, groups=groups
        )

    def predict_score(
        self, X: ArrayLike
    ) -> NDArray:
        """
        Compute the predicted probas, used to compute the
        conformity scores.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            Predicted probas
        """
        return self.predictor_.predict_proba(X)

    def predict_bounds(
        self,
        X: ArrayLike,
        y_pred: NDArray,
        **kwargs,
    ) -> NDArray:
        """
        Compute the prediction sets, using the fitted ``calibrator_``.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y_pred: 2D NDArray
            Observed Target

        z: ArrayLike
            Exogenous variables

        Returns
        -------
        NDArray
            Prediction sets, as a 3D array of shape (n_samples, n_classes, 1)
            for compatibility reason with ``MapieClassifier``.
        """
        # Classification conformity scores always have ``sym=True``, so
        # the calibrator_.predict result is a 2D array with
        # column 1 = -1 * column 2, So the true values are in res[:, 1]
        predict_kwargs = self._get_method_arguments(
            self.calibrator_.predict,
            dict(zip(["X", "y_pred"], [X, y_pred])),
            kwargs,
        )
        conformity_score_pred = self.calibrator_.predict(**predict_kwargs)

        self.conformity_score_ = cast(
            BaseClassificationScore, self.conformity_score_
        )

        self.conformity_score_.quantiles_ = conformity_score_pred[:, [1]][
            :, :, np.newaxis
        ]

        y_pred_set = self.conformity_score_.get_prediction_sets(
            y_pred_proba=y_pred[:, :, np.newaxis],
            conformity_scores=np.array([None]),  # never used in split
            alpha_np=np.array([self.alpha]),
            estimator=EnsembleClassifier(   # For compatibility. Only need cv
                self.predictor_,
                n_classes=len(np.unique(self.predictor_.classes_)),
                cv="prefit",
                n_jobs=-1,
                random_state=self.random_state,
                test_size=0.1,
                verbose=0,
            )
        )

        return y_pred_set

    def predict_best(self, y_pred: NDArray) -> NDArray:
        """
        Compute the prediction from the probas, using ``numpy.argmax``.

        Parameters
        ----------
        y_pred: NDArray
            Prediction scores (can be the prediction, the probas, ...)

        Returns
        -------
        NDArray
            best predictions
        """
        return np.argmax(y_pred, axis=1)
