from __future__ import annotations

from typing import Optional, Union, Tuple, Iterable
from typing_extensions import Self

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LogisticRegression

from mapie._typing import ArrayLike, NDArray
from mapie.classification import MapieClassifier
from mapie.conformity_scores import BaseClassificationScore
from mapie_v1._utils import (
    transform_confidence_level_to_alpha_list,
    prepare_params,
    cast_predictions_to_ndarray_tuple, cast_point_predictions_to_ndarray,
)
from mapie_v1.conformity_scores._utils import check_and_select_conformity_score


class SplitConformalClassifier:
    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        prefit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """
        Notes
        -----
        This implementation currently uses a ShuffleSplit cross-validation scheme
        for splitting the conformalization set. Future implementations may allow the use
        of groups.
        """
        self._estimator = estimator  # TODO: check
        self._alphas = transform_confidence_level_to_alpha_list(
            confidence_level
        )
        self._conformity_score = check_and_select_conformity_score(
            conformity_score,
            BaseClassificationScore
        )
        self._prefit = prefit

        # Note to developers: to implement this v1 class without touching the
        # v0 backend, we're for now using a hack. We always set cv="prefit",
        # and we fit the estimator if needed. See the .fit method below.
        self._mapie_classifier = MapieClassifier(
            estimator=self._estimator,
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self._conformity_score,
            random_state=random_state,
        )
        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        if not self._prefit:
            cloned_estimator = clone(self._estimator)
            fit_params_ = prepare_params(fit_params)
            cloned_estimator.fit(X_train, y_train, **fit_params_)
            self._mapie_classifier.estimator = cloned_estimator
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Specify that predict_params are passed to predict AND predict_proba,
        and are used in .conformalize but also in .predict_set and .predict
        """
        self._predict_params = prepare_params(predict_params)
        self._mapie_classifier.fit(
            X_conformalize,
            y_conformalize,
            predict_params=self._predict_params,
        )
        return self

    def predict_set(
        self,
        X: ArrayLike,
        conformity_score_params: Optional[dict] = None,
        # Prediction time parameters specific to conformity scores,
        # The only example for now is: include_last_label
        # Add the doc of include_last_label to the docstring
    ) -> Tuple[NDArray, NDArray]:
        """
        Shapes: (n, ) and (n, n_class, n_confidence_levels)
        """
        conformity_score_params_ = prepare_params(conformity_score_params)
        predictions = self._mapie_classifier.predict(
            X,
            alpha=self._alphas,
            include_last_label=conformity_score_params_.get("include_last_label", True),
            **self._predict_params,
        )
        return cast_predictions_to_ndarray_tuple(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        predictions = self._mapie_classifier.predict(
            X,
            alpha=None,
            **self._predict_params,
        )
        return cast_point_predictions_to_ndarray(predictions)


class CrossConformalClassifier:
    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        groups: Optional[ArrayLike] = None,
        predict_params: Optional[dict] = None
    ) -> Self:
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        return np.ndarray(0)

    def predict_set(
        self,
        X: ArrayLike,
        aggregation_method: Optional[str] = "mean",
        # How to aggregate the scores by the estimators on test data
        conformity_score_params: Optional[dict] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Shape: (n, ), (n, n_class, n_confidence_level)
        """
        return np.ndarray(0), np.ndarray(0)
