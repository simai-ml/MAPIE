from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union, cast
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict

from mapie._typing import ArrayLike, NDArray
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from mapie.utils import check_alpha

METHODS = ["score", "cumulated_score", "raps"]
WRONG_METHODS = ["scores", "cumulated", "test", "", 1, 2.5, (1, 2)]
WRONG_INCLUDE_LABELS = ["randomised", "True", "False", "other", 1, 2.5, (1, 2)]
Y_PRED_PROBA_WRONG = [
    np.array(
        [
            [0.8, 0.01, 0.1, 0.05],
            [1.0, 0.1, 0.0, 0.0]
        ]
    ),
    np.array(
        [
            [1.0, 0.0001, 0.0]
        ]
    ),
    np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.9, 0.01, 0.04, 0.06]
        ]
    ),
    np.array(
        [
            [0.8, 0.1, 0.02, 0.05],
            [0.9, 0.01, 0.03, 0.06]
        ]
    )
]

Y_TRUE_PROBA_PLACE = [
    [
        np.array([2, 0]),
        np.array([
            [.1, .3, .6],
            [.2, .7, .1]
        ]),
        np.array([[0], [1]])
    ],
    [
        np.array([1, 0]),
        np.array([
            [.7, .12, .18],
            [.5, .24, .26]
        ]),
        np.array([[2], [0]])
    ]
]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "cv": Optional[Union[int, str]],
        "test_size": Optional[Union[int, float]],
        "random_state": Optional[int]
    }
)
ParamsPredict = TypedDict(
    "ParamsPredict",
    {
        "include_last_label": Union[bool, str],
        "agg_scores": str
    }
)

STRATEGIES = {
    "score": (
        Params(
            method="score",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "score_split": (
        Params(
            method="score",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "score_cv_mean": (
        Params(
            method="score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "score_cv_crossval": (
        Params(
            method="score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="crossval"
        )
    ),
    "cumulated_score_include": (
        Params(
            method="cumulated_score",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "cumulated_score_not_include": (
        Params(
            method="cumulated_score",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "cumulated_score_randomized": (
        Params(
            method="cumulated_score",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="mean"
        )
    ),
    "cumulated_score_include_split": (
        Params(
            method="cumulated_score",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "cumulated_score_not_include_split": (
        Params(
            method="cumulated_score",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "cumulated_score_randomized_split": (
        Params(
            method="cumulated_score",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="mean"
        )
    ),
    "cumulated_score_include_cv_mean": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "cumulated_score_not_include_cv_mean": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="mean"
        )
    ),
    "cumulated_score_randomized_cv_mean": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="mean"
        )
    ),
    "cumulated_score_include_cv_crossval": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="crossval"
        )
    ),
    "cumulated_score_not_include_cv_crossval": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False,
            agg_scores="crossval"
        )
    ),
    "cumulated_score_randomized_cv_crossval": (
        Params(
            method="cumulated_score",
            cv=3,
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="crossval"
        )
    ),
    "naive": (
        Params(
            method="naive",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "naive_split": (
        Params(
            method="naive",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "top_k": (
        Params(
            method="top_k",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "top_k_split": (
        Params(
            method="top_k",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "raps": (
        Params(
            method="raps",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "raps_split": (
        Params(
            method="raps",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True,
            agg_scores="mean"
        )
    ),
    "raps_randomized": (
        Params(
            method="raps",
            cv="prefit",
            test_size=None,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="mean"
        )
    ),
    "raps_randomized_split": (
        Params(
            method="raps",
            cv="split",
            test_size=0.5,
            random_state=42
        ),
        ParamsPredict(
            include_last_label="randomized",
            agg_scores="mean"
        )
    ),
}

COVERAGES = {
    "score": 6/9,
    "score_split": 8/9,
    "score_cv_mean": 1.0,
    "score_cv_crossval": 1.0,
    "cumulated_score_include": 1.0,
    "cumulated_score_not_include": 5/9,
    "cumulated_score_randomized": 6/9,
    "cumulated_score_include_split": 8/9,
    "cumulated_score_not_include_split": 5/9,
    "cumulated_score_randomized_split": 7/9,
    "cumulated_score_include_cv_mean": 1.0,
    "cumulated_score_not_include_cv_mean": 5/9,
    "cumulated_score_randomized_cv_mean": 8/9,
    "cumulated_score_include_cv_crossval": 4/9,
    "cumulated_score_not_include_cv_crossval": 1/9,
    "cumulated_score_randomized_cv_crossval": 7/9,
    "naive": 5/9,
    "naive_split": 5/9,
    "top_k": 1.0,
    "top_k_split": 1.0,
    "raps": 1.0,
    "raps_split": 7/9,
    "raps_randomized": 8/9,
    "raps_randomized_split": 1.0
}

X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 1, 2, 1, 2])
y_toy_string = np.array(["0", "0", "1", "0", "1", "1", "2", "1", "2"])

y_toy_mapie = {
    "score": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, False, True]
    ],
    "score_split": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ],
    "score_cv_mean": [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "score_cv_crossval": [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "cumulated_score_include": [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, True, True],
        [False, False, True]
    ],
    "cumulated_score_not_include": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, True],
        [False, False, True]
    ],
    "cumulated_score_randomized": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, False, True]
    ],
    "cumulated_score_include_split": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, True],
        [True, True, True],
        [False, True, True],
        [False, False, True],
        [False, False, True]
    ],
    "cumulated_score_not_include_split": [
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [True, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, True],
        [False, False, True]
    ],
    "cumulated_score_randomized_split": [
        [False, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, True],
        [False, False, True]
    ],
    "cumulated_score_include_cv_mean": [
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "cumulated_score_not_include_cv_mean": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, True],
        [False, False, True]
    ],
    "cumulated_score_randomized_cv_mean": [
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, True, True]
    ],
    "cumulated_score_include_cv_crossval": [
        [False, False, False],
        [True, False, False],
        [False, False, False],
        [False, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, False]
    ],
    "cumulated_score_not_include_cv_crossval": [
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, True, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False]
    ],
    "cumulated_score_randomized_cv_crossval": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [True, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, False],
        [False, False, True]
    ],
    "naive": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, False, True],
        [False, False, True]
    ],
    "naive_split": [
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, True],
        [False, False, True]
    ],
    "top_k": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "top_k_split": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "raps": [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True]
    ],
    "raps_split": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False]
    ],
    "raps_randomized": [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, False, True]
    ],
    "raps_randomized_split": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True]
    ]
}

REGULARIZATION_PARAMETERS = [
    [.001, [1]],
    [[.01, .2], [1, 3]],
    [.1, [2, 4]]
]

IMAGE_INPUT = [
    {
        "X_calib": np.zeros((3, 1024, 1024, 1)),
        "X_test": np.ones((3, 1024, 1024, 1)),
    },
    {
        "X_calib": np.zeros((3, 512, 512, 3)),
        "X_test": np.ones((3, 512, 512, 3)),
    },
    {
        "X_calib": np.zeros((3, 256, 512)),
        "X_test": np.ones((3, 256, 512)),
    }
]

X_good_image = np.zeros((3, 1024, 1024, 3))
y_toy_image = np.array([0, 0, 1])

n_classes = 4
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_classes=n_classes,
    random_state=1,
)


class CumulatedScoreClassifier:

    def __init__(self) -> None:
        self.X_calib = np.array([0, 1, 2]).reshape(-1, 1)
        self.y_calib = np.array([0, 1, 2])
        self.y_calib_scores = np.array(
            [[0.750183952461055], [0.029571416154050345], [0.9268006058188594]]
        )
        self.X_test = np.array([3, 4, 5]).reshape(-1, 1)
        self.y_pred_sets = np.array(
            [
                [True, True, False],
                [False, True, False],
                [False, True, True],
                [True, True, False]
            ]
        )
        self.classes_ = self.y_calib

    def fit(self, X: ArrayLike, y: ArrayLike) -> CumulatedScoreClassifier:
        self.fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        return np.array([1, 2, 1])

    def predict_proba(self, X: ArrayLike) -> NDArray:
        if np.max(X) <= 2:
            return np.array(
                [[0.4, 0.5, 0.1], [0.2, 0.6, 0.2], [0.6, 0.3, 0.1]]
            )
        else:
            return np.array(
                [[0.2, 0.7, 0.1], [0., 1., 0.], [0., .7, 0.3], [0.3, .7, 0.]]
            )


class ImageClassifier:

    def __init__(self, X_calib: ArrayLike, X_test: ArrayLike) -> None:
        self.X_calib = X_calib
        self.y_calib = np.array([0, 1, 2])
        self.y_calib_scores = np.array(
            [[0.750183952461055], [0.029571416154050345], [0.9268006058188594]]
        )
        self.X_test = X_test
        self.y_pred_sets = np.array(
            [[True, True, False], [False, True, True], [True, True, False]]
        )
        self.classes_ = self.y_calib

    def fit(self, *args: Any) -> ImageClassifier:
        self.fitted_ = True
        return self

    def predict(self, *args: Any) -> NDArray:
        return np.array([1, 2, 1])

    def predict_proba(self, X: ArrayLike) -> NDArray:
        if np.max(X) == 0:
            return np.array(
                [[0.4, 0.5, 0.1], [0.2, 0.6, 0.2], [0.6, 0.3, 0.1]]
            )
        else:
            return np.array(
                [[0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]
            )


class WrongOutputModel:

    def __init__(self, proba_out: NDArray):
        self.trained_ = True
        self.proba_out = proba_out
        self.classes_ = np.arange(len(np.unique(proba_out[0])))

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict_proba(self, *args: Any) -> NDArray:
        return self.proba_out

    def predict(self, *args: Any) -> NDArray:
        pred = (
            self.proba_out == self.proba_out.max(axis=1)[:, None]
        ).astype(int)
        return pred


class Float32OuputModel:

    def __init__(self, prefit: bool = True):
        self.trained_ = prefit
        self.classes_ = [0, 1, 2]

    def fit(self, *args: Any) -> None:
        """Dummy fit."""
        self.trained_ = True

    def predict_proba(self, X: NDArray, *args: Any) -> NDArray:
        probas = np.array([[.9, .05, .05]])
        proba_out = np.repeat(probas, len(X), axis=0).astype(np.float32)
        return proba_out

    def predict(self, X: NDArray, *args: Any) -> NDArray:
        return np.repeat(1, len(X))

    def get_params(self, *args: Any, **kwargs: Any):
        return {"prefit": False}


def do_nothing(*args: Any) -> None:
    "Mock function that does nothing."
    pass


def test_mapie_classifier_sklearn_estim() -> None:
    """Test that MapieClassifier is an sklearn estimator"""
    check_estimator(MapieClassifier())


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieClassifier()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_clf = MapieClassifier()
    assert mapie_clf.method == "score"


@pytest.mark.parametrize("cv", ["prefit", "split"])
def test_warning_binary_classif(cv: str) -> None:
    """Test that a warning is raised y is binary."""
    mapie_clf = MapieClassifier(cv=cv, random_state=42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=3,
        n_classes=2,
        random_state=1,
    )
    with pytest.warns(UserWarning, match=r"not of type multiclass*"):
        mapie_clf.fit(X, y)


def test_binary_classif_same_result() -> None:
    """Test MAPIE doesnt change model output when y is binary."""
    mapie_clf = MapieClassifier(random_state=42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=3,
        n_classes=2,
        random_state=1,
    )
    mapie_predict = mapie_clf.fit(X, y).predict(X)
    lr = LogisticRegression(multi_class="multinomial").fit(X, y)
    lr_predict = lr.predict(X)
    np.testing.assert_allclose(mapie_predict, lr_predict)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    clf = LogisticRegression().fit(X_toy, y_toy)
    mapie_clf = MapieClassifier(estimator=clf, **STRATEGIES[strategy][0])
    mapie_clf.fit(X_toy, y_toy)
    assert isinstance(mapie_clf.single_estimator_, LogisticRegression)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = MapieClassifier(method=method, cv="prefit")
    mapie_clf.fit(X_toy, y_toy)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize(
    "cv", [None, -1, 2, KFold(), LeaveOneOut(),
           ShuffleSplit(n_splits=1, test_size=0.5), "prefit"]
)
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raises no errors."""
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_toy, y_toy)
    mapie_clf = MapieClassifier(estimator=model, cv=cv)
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(X_toy, alpha=0.5)


@pytest.mark.parametrize("agg_scores", ["mean", "crossval"])
def test_agg_scores_argument(agg_scores: str) -> None:
    """Test that predict passes with all valid 'agg_scores' arguments."""
    mapie_clf = MapieClassifier(cv=3, method="score")
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(X_toy, alpha=0.5, agg_scores=agg_scores)


@pytest.mark.parametrize("agg_scores", ["median", 1, None])
def test_invalid_agg_scores_argument(agg_scores: str) -> None:
    """Test that invalid 'agg_scores' raise errors."""
    mapie_clf = MapieClassifier(cv=3, method="score")
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(
        ValueError, match=r".*Invalid 'agg_scores' argument.*"
    ):
        mapie_clf.predict(X_toy, alpha=0.5, agg_scores=agg_scores)


@pytest.mark.parametrize("cv", [100, 200, 300])
def test_too_large_cv(cv: Any) -> None:
    """Test that too large cv raise sklearn errors."""
    mapie_clf = MapieClassifier(cv=cv)
    with pytest.raises(
        ValueError,
        match=rf".*Cannot have number of splits n_splits={cv} greater.*",
    ):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "include_last_label",
    [-3.14, 1.5, -2, 0, 1, "cv", DummyClassifier(), [1, 2]]
)
def test_invalid_include_last_label(include_last_label: Any) -> None:
    """Test that invalid include_last_label raise errors."""
    mapie_clf = MapieClassifier()
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(
        ValueError, match=r".*Invalid include_last_label argument.*"
    ):
        mapie_clf.predict(
            X_toy,
            y_toy,
            include_last_label=include_last_label
        )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_predict_output_shape(
    strategy: str, alpha: Any,
) -> None:
    """Test predict output shape."""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf = MapieClassifier(**args_init)
    mapie_clf.fit(X, y)
    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_ps.shape == (X.shape[0], len(np.unique(y)), n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_y_is_list_of_string(
    strategy: str, alpha: Any,
) -> None:
    """Test predict output shape with string y."""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf = MapieClassifier(**args_init)
    mapie_clf.fit(X, y.astype('str'))
    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_ps.shape == (X.shape[0], len(np.unique(y)), n_alpha)


@pytest.mark.parametrize(
    "strategy", ["naive", "top_k", "score", "cumulated_score_include"]
)
def test_same_results_prefit_split(strategy: str) -> None:
    """
    Test checking that if split and prefit method have exactly
    the same data split, then we have exactly the same results.
    """
    random_state = 1
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=3,
        n_classes=n_classes,
        random_state=random_state,
    )
    cv = ShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    train_index, val_index = next(cv.split(X))
    X_train_, X_calib_ = X[train_index], X[val_index]
    y_train_, y_calib_ = y[train_index], y[val_index]

    args_init, args_predict = deepcopy(STRATEGIES[strategy + '_split'])
    args_init["cv"] = cv
    mapie_reg = MapieClassifier(**args_init)
    mapie_reg.fit(X, y)
    y_pred_1, y_pis_1 = mapie_reg.predict(X, alpha=0.1, **args_predict)

    args_init, _ = STRATEGIES[strategy]
    model = LogisticRegression().fit(X_train_, y_train_)
    mapie_reg = MapieClassifier(estimator=model, **args_init)
    mapie_reg.fit(X_calib_, y_calib_)
    y_pred_2, y_pis_2 = mapie_reg.predict(X, alpha=0.1, **args_predict)

    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_same_result_y_numeric_and_string(
    strategy: str, alpha: Any,
) -> None:
    """Test that MAPIE outputs the same results if y is
    numeric or string"""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf_str = MapieClassifier(**args_init)
    mapie_clf_str.fit(X, y.astype('str'))
    mapie_clf_int = MapieClassifier(**args_init)
    mapie_clf_int.fit(X, y)
    _, y_ps_str = mapie_clf_str.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"],
    )
    _, y_ps_int = mapie_clf_int.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_ps_int, y_ps_str)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_y_1_to_l_minus_1(
    strategy: str, alpha: Any,
) -> None:
    """Test predict output shape with string y."""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf = MapieClassifier(**args_init)
    mapie_clf.fit(X, y + 1)
    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_ps.shape == (X.shape[0], len(np.unique(y)), n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_same_result_y_numeric_and_1_to_l_minus_1(
    strategy: str, alpha: Any,
) -> None:
    """Test that MAPIE outputs the same results if y is
    numeric or string"""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf_1 = MapieClassifier(**args_init)
    mapie_clf_1.fit(X, y + 1)
    mapie_clf_int = MapieClassifier(**args_init)
    mapie_clf_int.fit(X, y)
    _, y_ps_1 = mapie_clf_1.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"],
    )
    _, y_ps_int = mapie_clf_int.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_ps_int, y_ps_1)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf = MapieClassifier(**args_init)
    mapie_clf.fit(X, y)
    _, y_ps = mapie_clf.predict(
        X,
        alpha=[0.1, 0.1],
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_ps[:, 0, 0], y_ps[:, 0, 1])
    np.testing.assert_allclose(y_ps[:, 1, 0], y_ps[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf = MapieClassifier(**args_init)
    mapie_clf.fit(X, y)
    y_pred_float1, y_ps_float1 = mapie_clf.predict(
        X,
        alpha=alpha[0],
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    y_pred_float2, y_ps_float2 = mapie_clf.predict(
        X,
        alpha=alpha[1],
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    y_pred_array, y_ps_array = mapie_clf.predict(
        X,
        alpha=alpha,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_ps_float1[:, :, 0], y_ps_array[:, :, 0])
    np.testing.assert_allclose(y_ps_float2[:, :, 0], y_ps_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    args_init, args_predict = STRATEGIES[strategy]
    mapie_clf_single = MapieClassifier(n_jobs=1, **args_init)
    mapie_clf_multi = MapieClassifier(n_jobs=-1, **args_init)
    mapie_clf_single.fit(X, y)
    mapie_clf_multi.fit(X, y)
    y_pred_single, y_ps_single = mapie_clf_single.predict(
        X,
        alpha=0.2,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    y_pred_multi, y_ps_multi = mapie_clf_multi.predict(
        X,
        alpha=0.2,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_ps_single, y_ps_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(
    strategy: str
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    args_init, args_predict = STRATEGIES[strategy]
    lr = LogisticRegression(C=1e-99)
    lr.fit(X, y)
    n_samples = len(X)
    mapie_clf0 = MapieClassifier(lr, **args_init)
    mapie_clf1 = MapieClassifier(lr, **args_init)
    mapie_clf2 = MapieClassifier(lr, **args_init)
    mapie_clf0.fit(X, y, sample_weight=None)
    mapie_clf1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie_clf2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_ps0 = mapie_clf0.predict(
        X,
        alpha=0.2,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    y_pred1, y_ps1 = mapie_clf1.predict(
        X,
        alpha=0.2,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    y_pred2, y_ps2 = mapie_clf2.predict(
        X,
        alpha=0.2,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred0, y_pred2)
    np.testing.assert_allclose(y_ps0, y_ps1)
    np.testing.assert_allclose(y_ps0, y_ps2)


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None]
)
def test_valid_prediction(alpha: Any) -> None:
    """Test fit and predict."""
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_toy, y_toy)
    mapie_clf = MapieClassifier(estimator=model, cv="prefit")
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_toy_dataset_predictions(strategy: str) -> None:
    """Test prediction sets estimated by MapieClassifier on a toy dataset"""
    args_init, args_predict = STRATEGIES[strategy]
    if "split" not in strategy:
        clf = LogisticRegression().fit(X_toy, y_toy)
    else:
        clf = LogisticRegression()
    mapie_clf = MapieClassifier(estimator=clf, **args_init)
    mapie_clf.fit(X_toy, y_toy, size_raps=.5)
    _, y_ps = mapie_clf.predict(
        X_toy,
        alpha=0.5,
        include_last_label=args_predict["include_last_label"],
        agg_scores=args_predict["agg_scores"]
    )
    np.testing.assert_allclose(y_ps[:, :, 0], y_toy_mapie[strategy])
    np.testing.assert_allclose(
        classification_coverage_score(y_toy, y_ps[:, :, 0]),
        COVERAGES[strategy],
    )


def test_cumulated_scores() -> None:
    """Test cumulated score method on a tiny dataset."""
    alpha = [0.65]
    quantile = [0.750183952461055]
    # fit
    cumclf = CumulatedScoreClassifier()
    cumclf.fit(cumclf.X_calib, cumclf.y_calib)
    mapie_clf = MapieClassifier(
        cumclf,
        method="cumulated_score",
        cv="prefit",
        random_state=42
    )
    mapie_clf.fit(cumclf.X_calib, cumclf.y_calib)
    np.testing.assert_allclose(
        mapie_clf.conformity_scores_, cumclf.y_calib_scores
    )
    # predict
    _, y_ps = mapie_clf.predict(
        cumclf.X_test,
        include_last_label=True,
        alpha=alpha
    )
    np.testing.assert_allclose(mapie_clf.quantiles_, quantile)
    np.testing.assert_allclose(y_ps[:, :, 0], cumclf.y_pred_sets)


@pytest.mark.parametrize("X", IMAGE_INPUT)
def test_image_cumulated_scores(X: Dict[str, ArrayLike]) -> None:
    """Test image as input for cumulated_score method."""
    alpha = [0.65]
    quantile = [0.750183952461055]
    # fit
    X_calib = X["X_calib"]
    X_test = X["X_test"]
    cumclf = ImageClassifier(X_calib, X_test)
    cumclf.fit(cumclf.X_calib, cumclf.y_calib)
    mapie = MapieClassifier(
        cumclf,
        method="cumulated_score",
        cv="prefit",
        random_state=42
    )
    mapie.fit(cumclf.X_calib, cumclf.y_calib)
    np.testing.assert_allclose(mapie.conformity_scores_, cumclf.y_calib_scores)
    # predict
    _, y_ps = mapie.predict(
        cumclf.X_test,
        include_last_label=True,
        alpha=alpha
    )
    np.testing.assert_allclose(mapie.quantiles_, quantile)
    np.testing.assert_allclose(y_ps[:, :, 0], cumclf.y_pred_sets)


@pytest.mark.parametrize("y_pred_proba", Y_PRED_PROBA_WRONG)
def test_sum_proba_to_one_fit(y_pred_proba: NDArray) -> None:
    """
    Test if when the output probabilities of the model do not
    sum to one, return an error in the fit method.
    """
    wrong_model = WrongOutputModel(y_pred_proba)
    mapie_clf = MapieClassifier(wrong_model, cv="prefit")
    with pytest.raises(
        AssertionError, match=r".*The sum of the scores is not equal to one.*"
    ):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize("y_pred_proba", Y_PRED_PROBA_WRONG)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_sum_proba_to_one_predict(
    y_pred_proba: NDArray,
    alpha: Union[float, Iterable[float]]
) -> None:
    """
    Test if when the output probabilities of the model do not
    sum to one, return an error in the predict method.
    """
    wrong_model = WrongOutputModel(y_pred_proba)
    mapie_clf = MapieClassifier(cv="prefit")
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.single_estimator_ = wrong_model
    with pytest.raises(
        AssertionError, match=r".*The sum of the scores is not equal to one.*"
    ):
        mapie_clf.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(), make_pipeline(LogisticRegression())]
)
def test_classifier_without_classes_attribute(
    estimator: ClassifierMixin
) -> None:
    """
    Test that prefitted classifier without 'classes_ 'attribute raises error.
    """
    estimator.fit(X_toy, y_toy)
    if isinstance(estimator, Pipeline):
        delattr(estimator[-1], "classes_")
    else:
        delattr(estimator, "classes_")
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    with pytest.raises(
        AttributeError, match=r".*does not contain 'classes_'.*"
    ):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
def test_method_error_in_fit(monkeypatch: Any, method: str) -> None:
    """Test else condition for the method in .fit"""
    monkeypatch.setattr(
        MapieClassifier, "_check_parameters", do_nothing
    )
    mapie_clf = MapieClassifier(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_method_error_in_predict(method: Any, alpha: float) -> None:
    """Test else condition for the method in .predict"""
    mapie_clf = MapieClassifier(method="score")
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.method = method
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie_clf.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize("include_labels", WRONG_INCLUDE_LABELS)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_include_label_error_in_predict(
    monkeypatch: Any, include_labels: Union[bool, str], alpha: float
) -> None:
    """Test else condition for include_label parameter in .predict"""
    monkeypatch.setattr(
        MapieClassifier,
        "_check_include_last_label",
        do_nothing
    )
    mapie_clf = MapieClassifier(method="cumulated_score")
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid include.*"):
        mapie_clf.predict(
            X_toy, alpha=alpha,
            include_last_label=include_labels
        )


def test_pred_loof_isnan() -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_clf = MapieClassifier()
    _, y_pred, _, _ = mapie_clf._fit_and_predict_oof_model(
        estimator=LogisticRegression(),
        X=X_toy,
        y=y_toy,
        train_index=[0, 1, 2, 3, 4],
        val_index=[],
        k=0,
    )
    assert len(y_pred) == 0


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_pipeline_compatibility(strategy: str) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 0])
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"])
        ]
    )
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(X, y)
    mapie = MapieClassifier(estimator=pipe, **STRATEGIES[strategy][0])
    mapie.fit(X, y)
    mapie.predict(X)


def test_pred_proba_float64() -> None:
    """Check that the method _check_proba_normalized returns float64."""
    y_pred_proba = np.random.random((1000, 10)).astype(np.float32)
    sum_of_rows = y_pred_proba.sum(axis=1)
    normalized_array = y_pred_proba / sum_of_rows[:, np.newaxis]
    mapie = MapieClassifier()
    checked_normalized_array = mapie._check_proba_normalized(normalized_array)

    assert checked_normalized_array.dtype == "float64"


@pytest.mark.parametrize("cv", ["prefit", None])
def test_classif_float32(cv) -> None:
    """
    Check that by returning float64 arrays there are not
    empty predictions sets with naive method using both
    prefit and cv=5. If the y_pred_proba was still in
    float32, as the quantile=0.90 would have been equal
    to the highest probability, MAPIE would have return
    empty prediction sets"""
    X_cal, y_cal = make_classification(
        n_samples=20,
        n_features=20,
        n_redundant=0,
        n_informative=20,
        n_classes=3
    )
    X_test, _ = make_classification(
        n_samples=20,
        n_features=20,
        n_redundant=0,
        n_informative=20,
        n_classes=3
    )
    alpha = .9
    dummy_classif = Float32OuputModel()

    mapie = MapieClassifier(
        estimator=dummy_classif, method="naive",
        cv=cv, random_state=42
    )
    mapie.fit(X_cal, y_cal)
    _, yps = mapie.predict(X_test, alpha=alpha, include_last_label=True)

    assert (
        np.repeat([[True, False, False]], 20, axis=0)[:, :, np.newaxis] == yps
    ).all()


@pytest.mark.parametrize("k_lambda", REGULARIZATION_PARAMETERS)
def test_regularize_conf_scores_shape(k_lambda) -> None:
    """
    Test that the conformity scores have the correct shape.
    """
    lambda_, k = k_lambda[0], k_lambda[1]
    args_init, _ = STRATEGIES["raps"]
    clf = LogisticRegression().fit(X, y)
    mapie_clf = MapieClassifier(estimator=clf, **args_init)
    conf_scores = np.random.rand(100, 1)
    cutoff = np.cumsum(np.ones(conf_scores.shape)) - 1
    reg_conf_scores = mapie_clf._regularize_conformity_score(
        k, lambda_, conf_scores, cutoff
    )

    assert reg_conf_scores.shape == (100, 1, len(k))


def test_get_true_label_cumsum_proba_shape() -> None:
    """
    Test that the true label cumsumed probabilities
    have the correct shape.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)
    mapie_clf = MapieClassifier(estimator=clf)
    mapie_clf.fit(X, y)
    cumsum_proba, cutoff = mapie_clf._get_true_label_cumsum_proba(
        y, y_pred
    )
    assert cumsum_proba.shape == (len(X), 1)
    assert cutoff.shape == (len(X), )


def test_get_true_label_cumsum_proba_result() -> None:
    """
    Test that the true label cumsumed probabilities
    are the expected ones.
    """
    clf = LogisticRegression()
    clf.fit(X_toy, y_toy)
    y_pred = clf.predict_proba(X_toy)
    mapie_clf = MapieClassifier(estimator=clf)
    mapie_clf.fit(X_toy, y_toy)
    cumsum_proba, cutoff = mapie_clf._get_true_label_cumsum_proba(
        y_toy, y_pred
    )
    np.testing.assert_allclose(
        cumsum_proba,
        np.array(
            [
                y_pred[0, 0], y_pred[1, 0],
                y_pred[2, 0] + y_pred[2, 1],
                y_pred[3, 0] + y_pred[3, 1],
                y_pred[4, 1], y_pred[5, 1],
                y_pred[6, 1] + y_pred[6, 2],
                y_pred[7, 1] + y_pred[7, 2],
                y_pred[8, 2]
            ]
        )[:, np.newaxis]
    )
    np.testing.assert_allclose(cutoff, np.array([1, 1, 2, 2, 1, 1, 2, 2, 1]))


@pytest.mark.parametrize("k_lambda", REGULARIZATION_PARAMETERS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_get_last_included_proba_shape(k_lambda, strategy):
    """
    Test that the outputs of _get_last_included_proba method
    have the correct shape.
    """
    lambda_, k = k_lambda[0], k_lambda[1]
    if len(k) == 1:
        thresholds = .2
    else:
        thresholds = np.random.rand(len(k))
    thresholds = cast(NDArray, check_alpha(thresholds))
    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)
    y_pred_proba = np.repeat(
        y_pred_proba[:, :, np.newaxis], len(thresholds), axis=2
    )

    mapie = MapieClassifier(estimator=clf, **STRATEGIES[strategy][0])
    include_last_label = STRATEGIES[strategy][1]["include_last_label"]
    y_p_p_c, y_p_i_l, y_p_p_i_l = mapie._get_last_included_proba(
        y_pred_proba, thresholds,
        include_last_label, lambda_, k
    )

    assert y_p_p_c.shape == (len(X), len(np.unique(y)), len(thresholds))
    assert y_p_i_l.shape == (len(X), 1, len(thresholds))
    assert y_p_p_i_l.shape == (len(X), 1, len(thresholds))


@pytest.mark.parametrize("y_true_proba_place", Y_TRUE_PROBA_PLACE)
def test_get_true_label_position(
    y_true_proba_place: List[NDArray]
) -> None:
    """
    Check that the returned true label position the good.
    """
    y_true = y_true_proba_place[0]
    y_pred_proba = y_true_proba_place[1]
    place = y_true_proba_place[2]

    mapie = MapieClassifier()
    found_place = mapie._get_true_label_position(y_pred_proba, y_true)

    assert (found_place == place).all()


@pytest.mark.parametrize("cv", [5, None])
def test_error_raps_cv_not_prefit(cv: Union[int, None]) -> None:
    """
    Test that an error is raised if the method is RAPS
    and cv is different from prefit and split.
    """
    mapie = MapieClassifier(method="raps", cv=5)
    with pytest.raises(ValueError, match=r".*RAPS method can only.*"):
        mapie.fit(X_toy, y_toy)


def test_not_all_label_in_calib() -> None:
    """
    Test that the true label cumsumed probabilities
    have the correct shape.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    indices_remove = np.where(y != 2)
    X_mapie = X[indices_remove]
    y_mapie = y[indices_remove]
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    mapie_clf.fit(X_mapie, y_mapie)
    y_pred, y_pss = mapie_clf.predict(X, alpha=0.5)
    assert y_pred.shape == (len(X), )
    assert y_pss.shape == (len(X), len(np.unique(y)), 1)


def test_warning_not_all_label_in_calib() -> None:
    """
    Test that a warning is raised y is binary.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    indices_remove = np.where(y != 2)
    X_mapie = X[indices_remove]
    y_mapie = y[indices_remove]
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    with pytest.warns(
        UserWarning, match=r".*WARNING: your calibration dataset.*"
    ):
        mapie_clf.fit(X_mapie, y_mapie)


def test_n_classes_prefit() -> None:
    """
    Test that the attribute n_classes_ has the correct
    value with cv="prefit".
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    indices_remove = np.where(y != 2)
    X_mapie = X[indices_remove]
    y_mapie = y[indices_remove]
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    mapie_clf.fit(X_mapie, y_mapie)
    assert mapie_clf.n_classes_ == len(np.unique(y))


def test_classes_prefit() -> None:
    """
    Test that the attribute classes_ has the correct
    value with cv="prefit".
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    indices_remove = np.where(y != 2)
    X_mapie = X[indices_remove]
    y_mapie = y[indices_remove]
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    mapie_clf.fit(X_mapie, y_mapie)
    assert (mapie_clf.classes_ == np.unique(y)).all()


def test_classes_encoder_same_than_model() -> None:
    """
    Test that the attribute label encoder has the same
    classes as the prefit model
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    indices_remove = np.where(y != 2)
    X_mapie = X[indices_remove]
    y_mapie = y[indices_remove]
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    mapie_clf.fit(X_mapie, y_mapie)
    assert (mapie_clf.label_encoder_.classes_ == np.unique(y)).all()


def test_n_classes_cv() -> None:
    """
    Test that the attribute n_classes_ has the correct
    value with cross_validation.
    """
    clf = LogisticRegression()

    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv=5
    )
    mapie_clf.fit(X, y)
    assert mapie_clf.n_classes_ == len(np.unique(y))


def test_classes_cv() -> None:
    """
    Test that the attribute classes_ has the correct
    value with cross_validation.
    """
    clf = LogisticRegression()

    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv=5
    )
    mapie_clf.fit(X, y)
    assert (mapie_clf.classes_ == np.unique(y)).all()


def test_raise_error_new_class() -> None:
    """
    Test that the attribute if there is an unseen
    classe in `y` then an error is raised.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    y[-1] = 10
    mapie_clf = MapieClassifier(
        estimator=clf, method="cumulated_score",
        cv="prefit"
    )
    with pytest.raises(
        ValueError, match=r".*Values in y do not matched values.*"
    ):
        mapie_clf.fit(X, y)
