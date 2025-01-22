from __future__ import annotations

from typing import Optional, Union, List
from typing_extensions import Self

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LogisticRegression

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseClassificationScore


class SplitConformalClassifier:
    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        prefit: bool = True,
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
        predict_params: Optional[dict] = None,
    ) -> Self:
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Return
        -----
        Return ponctual prediction similar to predict method of
        scikit-learn classifiers
        Shape (n_samples,)
        """
        return np.ndarray(0)

    def predict_sets(
        self,
        X: ArrayLike,
        conformity_score_params: Optional[dict] = None,
        # Parameters specific to conformal method,
        # For example: include_last_label
    ) -> NDArray:
        """
        Return
        -----
        An array containing the prediction sets
        Shape (n_samples, n_classes) if confidence_level is float,
        Shape (n_samples, n_classes, confidence_level) if confidence_level
        is a list of floats
        """
        return np.ndarray(0)


class CrossConformalClassifier:
    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        cross_val: Union[BaseCrossValidator, str] = 5,
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
        predict_params: Optional[dict] = None
    ) -> Self:
        return self

    def predict(self,
                X: ArrayLike) -> NDArray:
        """
        Return
        -----
        Return ponctual prediction similar to predict method of
        scikit-learn classifiers
        Shape (n_samples,)
        """
        return np.ndarray(0)

    def predict_sets(
        self,
        X: ArrayLike,
        aggregation_method: Optional[str] = "mean",
        # How to aggregate the scores by the estimators on test data
        conformity_score_params: Optional[dict] = None
    ) -> NDArray:
        """
        Return
        -----
        An array containing the prediction sets
        Shape (n_samples, n_classes) if confidence_level is float,
        Shape (n_samples, n_classes, confidence_level) if confidence_level
        is a list of floats
        """
        return np.ndarray(0)
