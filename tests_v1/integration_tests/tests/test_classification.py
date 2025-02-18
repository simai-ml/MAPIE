import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    RAPSConformityScore,
    APSConformityScore,
    TopKConformityScore, LACConformityScore,
)
from mapie_v1.classification import SplitConformalClassifier
from tests_v1.integration_tests.utils import (
    DummyClassifierWithFitAndPredictParams,
    train_test_split_shuffle,
)
from mapie._typing import ArrayLike

pytestmark = pytest.mark.classification

RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=1000,
        n_informative=5,
        n_classes=4,
        random_state=RANDOM_STATE
    )
    sample_weight = RandomState(RANDOM_STATE).random(len(X))

    (
        X_train,
        X_conformalize,
        y_train,
        y_conformalize,
        sample_weight_train,
        sample_weight_conformalize,
    ) = train_test_split_shuffle(
        X, y, random_state=RANDOM_STATE, sample_weight=sample_weight
    )

    return {
        "X": X,
        "y": y,
        "sample_weight": sample_weight,
        "X_train": X_train,
        "X_conformalize": X_conformalize,
        "y_train": y_train,
        "y_conformalize": y_conformalize,
        "sample_weight_train": sample_weight_train,
        "sample_weight_conformalize": sample_weight_conformalize,
    }


@pytest.fixture()
def params_split_test_1():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
            },
        },
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": LACConformityScore(),
                "cv": "prefit"
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.fixture()
def params_split_test_2():
    return {
        "v1": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "confidence_level": 0.8,
                "prefit": False,
                "conformity_score": "top_k",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"dummy_fit_param": True},
            },
            "conformalize": {
                "predict_params": {"dummy_predict_param": True},
            }},
        "v0": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "conformity_score": TopKConformityScore(),
                "cv": "split",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"dummy_fit_param": True},
                "predict_params": {"dummy_predict_param": True},
            },
            "predict": {
                "alpha": 0.2,
                "dummy_predict_param": True,
            }}}


@pytest.fixture()
def params_split_test_3(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "confidence_level": [0.8, 0.9],
                "prefit": False,
                "conformity_score": "aps",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"sample_weight": dataset["sample_weight_train"]},
            },
            "predict_set": {
                "conformity_score_params": {"include_last_label": False}
            }},
        "v0": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "conformity_score": APSConformityScore(),
                "cv": "split",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "sample_weight": dataset["sample_weight"],
            },
            "predict": {
                "alpha": [0.2, 0.1],
                "include_last_label": False,
            }}}


@pytest.fixture()
def params_split_test_4():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": "raps",
                "random_state": RANDOM_STATE,
            }},
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": RAPSConformityScore(),
                "cv": "prefit",
                "random_state": RANDOM_STATE,
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.mark.parametrize(
    "params", [
        "params_split_test_1",
        "params_split_test_2",
        "params_split_test_3",
        "params_split_test_4",
    ]
)
def test_intervals_and_predictions_exact_equality_split(dataset, params, request):
    X, y, X_train, X_conformalize, y_train, y_conformalize = (
        dataset["X"],
        dataset["y"],
        dataset["X_train"],
        dataset["X_conformalize"],
        dataset["y_train"],
        dataset["y_conformalize"],
    )

    params_ = request.getfixturevalue(params)

    v0_init_params = params_["v0"].get("__init__", {})
    v0_fit_params = params_["v0"].get("fit", {})
    v0_predict_params = params_["v0"].get("predict", {})

    v1_init_params = params_["v1"].get("__init__", {})
    v1_fit_params = params_["v1"].get("fit", {})
    v1_conformalize_params = params_["v1"].get("conformalize", {})
    v1_predict_set_params = params_["v1"].get("predict_set", {})

    prefit = v1_init_params.get("prefit", True)

    if prefit:
        v0_init_params["estimator"].fit(X_train, y_train)
        v1_init_params["estimator"].fit(X_train, y_train)

    v0 = MapieClassifier(**v0_init_params)
    v1 = SplitConformalClassifier(**v1_init_params)

    if prefit:
        v0.fit(X_conformalize, y_conformalize, **v0_fit_params)
    else:
        v0.fit(X, y, **v0_fit_params)
        v1.fit(X_train, y_train, **v1_fit_params)
    v1.conformalize(X_conformalize, y_conformalize, **v1_conformalize_params)

    v0_preds, v0_pred_sets = v0.predict(X_conformalize, **v0_predict_params)
    v1_preds, v1_pred_sets = v1.predict_set(X_conformalize, **v1_predict_set_params)

    v1_preds_using_predict: ArrayLike = v1.predict(X_conformalize)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_sets, v1_pred_sets)
    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)

    confidence_level = v1_init_params.get("confidence_level", 0.9)
    n_confidence_level = 1 if isinstance(confidence_level, float) else len(
        confidence_level
    )
    assert v1_pred_sets.shape == (
        len(X_conformalize),
        len(np.unique(y)),
        n_confidence_level,
    )
