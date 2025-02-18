from typing import Callable, Dict, Any, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing_extensions import Self

from mapie._typing import NDArray, ArrayLike
import inspect
from sklearn.model_selection import ShuffleSplit


def train_test_split_shuffle(
    X: NDArray,
    y: NDArray,
    test_size: float = None,
    random_state: int = 42,
    sample_weight: Optional[NDArray] = None,
) -> Union[Tuple[Any, Any, Any, Any], Tuple[Any, Any, Any, Any, Any, Any]]:
    splitter = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if sample_weight is not None:
        sample_weight_train = sample_weight[train_idx]
        sample_weight_test = sample_weight[test_idx]
        return X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test

    return X_train, X_test, y_train, y_test


def filter_params(
    function: Callable,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if params is None:
        return {}

    model_params = inspect.signature(function).parameters
    return {k: v for k, v in params.items() if k in model_params}


class DummyClassifierWithFitAndPredictParams(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self._dummy_fit_param = None

    def fit(self, X: ArrayLike, y: ArrayLike, dummy_fit_param: bool = False) -> Self:
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Dummy classifier needs at least 3 classes")
        self._dummy_fit_param = dummy_fit_param
        return self

    def predict_proba(self, X: ArrayLike, dummy_predict_param: bool = False) -> NDArray:
        probas = np.zeros((len(X), len(self.classes_)))
        if self._dummy_fit_param & dummy_predict_param:
            probas[:, 0] = 0.1
            probas[:, 1] = 0.9
        elif self._dummy_fit_param:
            probas[:, 1] = 0.1
            probas[:, 2] = 0.9
        elif dummy_predict_param:
            probas[:, 1] = 0.1
            probas[:, 0] = 0.9
        else:
            probas[:, 2] = 0.1
            probas[:, 0] = 0.9
        return probas

    def predict(self, X: ArrayLike, dummy_predict_param: bool = False) -> NDArray:
        y_preds_proba = self.predict_proba(X, dummy_predict_param)
        return np.amax(y_preds_proba, axis=0)
