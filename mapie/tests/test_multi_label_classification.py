from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.multi_label_classification import MapieMultiLabelClassifier

Params = TypedDict(
    "Params",
    {
        "method": str,
        "bound": Optional[str],
        "random_state": Optional[int]
    }
)

METHODS = ["crc", "rcps"]
BOUNDS = ["wsr", "hoeffding", "bernstein"]

WRONG_METHODS = ["rpcs", "rcr", "test", "", 1, 2.5, (1, 2)]
WRONG_BOUNDS = ["wrs", "hoeff", "test", "", 1, 2.5, (1, 2)]


STRATEGIES = {
    "crc": (
        Params(
            method="crc",
            bound=None,
            random_state=42
        ),
    ),
    "rcps_wsr": (
        Params(
            method="rcps",
            bound="wsr",
            random_state=42
        ),
    ),
    "rcps_hoeffding": (
        Params(
            method="rcps",
            bound="hoeffding",
            random_state=42
        ),
    ),
    "rcps_bernstein": (
        Params(
            method="rcps",
            bound="bernstein",
            random_state=42
        ),
    ),
}


y_toy_mapie = {
    "crc": [
        [True, False, True],
        [True, False, True],
        [True, False, True],
        [True, False, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [False, True, True]
    ],
    "rcps_bernstein": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
    "rcps_wsr": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
    "rcps_hoeffding": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
}


class WrongOutputModel:

    def __init__(self):
        pass

    def predict_proba(self, *args: Any) -> NDArray:
        """Dummy predict_proba."""

    def predict(self, *args: Any) -> NDArray:
        """Dummy predict."""


class ArrayOutputModel:

    def __init__(self):
        self.trained_ = True

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict_proba(self, X: NDArray, *args: Any) -> NDArray:
        probas = np.array([[.9, .05, .05]])
        proba_out = np.repeat(probas, len(X), axis=0)
        return proba_out

    def predict(self, X: NDArray, *args: Any) -> NDArray:
        return self.predict_proba(X) >= .3

    def __sklearn_is_fitted__(self):
        return True


X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.stack(
    [
        [1, 0, 1], [1, 0, 0], [0, 1, 1],
        [0, 1, 0], [0, 0, 1], [1, 1, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
)

X, y = make_multilabel_classification(
    n_samples=1000,
    n_classes=5,
    random_state=42,
    allow_unlabeled=False
)

X_no_label, y_no_label = make_multilabel_classification(
    n_samples=1000,
    n_classes=5,
    random_state=42,
    allow_unlabeled=True
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieMultiLabelClassifier()


def test_valid_estimator() -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = MapieMultiLabelClassifier(estimator=clf)
    mapie_clf.fit(X_toy, y_toy)
    assert isinstance(mapie_clf.single_estimator_, MultiOutputClassifier)


def test_valid_method() -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize("bound", BOUNDS)
def test_valid_bound(bound: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(X_toy, method="rcps", bound=bound, delta=.1)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
@pytest.mark.parametrize("delta", [0.2, 0.1, 0.05])
def test_predict_output_shape(
    strategy: str, alpha: Any, delta: Any
) -> None:
    """Test predict output shape."""
    args = STRATEGIES[strategy][0]
    mapie_clf = MapieMultiLabelClassifier()
    mapie_clf.fit(X, y)
    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=alpha,
        method=args["method"],
        bound=args["bound"],
        delta=.1
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == y.shape
    assert y_ps.shape == (y.shape[0], y.shape[1], n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf = MapieMultiLabelClassifier()
    mapie_clf.fit(X, y)
    _, y_ps = mapie_clf.predict(
        X,
        alpha=[0.1, 0.1],
        method=args["method"],
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(y_ps[:, 0, 0], y_ps[:, 0, 1])
    np.testing.assert_allclose(y_ps[:, 1, 0], y_ps[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_partial_fit(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    mapie_clf = MapieMultiLabelClassifier(clf)
    mapie_clf.fit(X, y)

    mapie_clf_partial = MapieMultiLabelClassifier(clf)
    for i in range(len(X)):
        mapie_clf_partial.partial_fit(
            X[i][np.newaxis, :],
            y[i][np.newaxis, :]
        )

    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=[0.1, 0.1],
        method=args["method"],
        bound=args["bound"],
        delta=.1
    )

    y_pred_partial, y_ps_partial = mapie_clf_partial.predict(
        X,
        alpha=[0.1, 0.1],
        method=args["method"],
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(y_pred, y_pred_partial)
    np.testing.assert_allclose(y_ps, y_ps_partial)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    args = STRATEGIES[strategy][0]
    mapie_clf = MapieMultiLabelClassifier()
    mapie_clf.fit(X, y)
    y_pred_float1, y_ps_float1 = mapie_clf.predict(
        X,
        alpha=alpha[0],
        method=args["method"],
        bound=args["bound"],
        delta=.9
    )
    y_pred_float2, y_ps_float2 = mapie_clf.predict(
        X,
        alpha=alpha[1],
        method=args["method"],
        bound=args["bound"],
        delta=.9
    )
    y_pred_array, y_ps_array = mapie_clf.predict(
        X,
        alpha=alpha,
        method=args["method"],
        bound=args["bound"],
        delta=.9
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
    args = STRATEGIES[strategy][0]
    mapie_clf_single = MapieMultiLabelClassifier(
        n_jobs=1,
        random_state=args["random_state"]
    )
    mapie_clf_multi = MapieMultiLabelClassifier(
        n_jobs=-1,
        random_state=args["random_state"]
    )
    mapie_clf_single.fit(X, y)
    mapie_clf_multi.fit(X, y)
    y_pred_single, y_ps_single = mapie_clf_single.predict(
        X,
        alpha=0.2,
        method=args["method"],
        bound=args["bound"],
        delta=.9
    )
    y_pred_multi, y_ps_multi = mapie_clf_multi.predict(
        X,
        alpha=0.2,
        method=args["method"],
        bound=args["bound"],
        delta=.9
    )
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_ps_single, y_ps_multi)


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None],
)
@pytest.mark.parametrize(
    "delta", [.1, .2, .5, .9, .001],
)
@pytest.mark.parametrize(
    "bound", BOUNDS,
)
def test_valid_prediction(alpha: Any, delta: Any, bound: Any) -> None:
    """Test fit and predict."""
    model = MultiOutputClassifier(
        LogisticRegression(multi_class="multinomial")
    )
    model.fit(X_toy, y_toy)
    mapie_clf = MapieMultiLabelClassifier(estimator=model)
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(
        X_toy,
        alpha=alpha,
        method="rcps",
        bound=bound,
        delta=delta
    )


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None],
)
@pytest.mark.parametrize(
    "delta", [.1, .2, .5, .9, .001],
)
@pytest.mark.parametrize(
    "bound", BOUNDS,
)
@pytest.mark.parametrize(
    "method", METHODS,
)
def test_array_output_model(method: Any, alpha: Any, delta: Any, bound: Any):
    model = ArrayOutputModel()
    mapie_clf = MapieMultiLabelClassifier(estimator=model)
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(
        X_toy,
        alpha=alpha,
        method=method,
        bound=bound,
        delta=delta
    )


def test_reinit_new_fit():
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = MapieMultiLabelClassifier(clf)
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.fit(X_toy, y_toy)
    assert len(mapie_clf.risks) == len(X_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
def test_method_error_in_predict(method: str) -> None:
    """Test error for wrong method"""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie_clf.predict(X_toy, method=method)


def test_method_error_if_no_label_fit() -> None:
    """Test error for wrong method"""
    mapie_clf = MapieMultiLabelClassifier()
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.fit(X_no_label, y_no_label)


def test_method_error_if_no_label_partial_fit() -> None:
    """Test error for wrong method"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(
        X_no_label,
        y_no_label
    )
    mapie_clf = MapieMultiLabelClassifier(clf)
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.partial_fit(X_no_label, y_no_label)


@pytest.mark.parametrize("bound", WRONG_BOUNDS)
def test_bound_error_in_predict(bound: str) -> None:
    """Test error for wrong method"""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*bound must be in.*"):
        mapie_clf.predict(X_toy, method="rcps", bound=bound, delta=.1)


def test_error_rcps_delta_null() -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta cannot be ``None``*"):
        mapie_clf.predict(X_toy, method="rcps")


@pytest.mark.parametrize("delta", [-1., 0, 1, 4, -3])
def test_error_delta_wrong_value(delta: Any) -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be*"):
        mapie_clf.predict(X_toy, method="rcps", delta=delta)


def test_bound_none_crc() -> None:
    """Test that a warning is raised nound is not None with CRC method."""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        mapie_clf.predict(X_toy, method="crc", bound="wsr")


def test_delta_none_crc() -> None:
    """Test that a warning is raised nound is not None with CRC method."""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        mapie_clf.predict(X_toy, method="crc", bound=None, delta=.1)


def test_warning_estimator_none() -> None:
    """Test that a warning is raised nound is not None with CRC method."""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    with pytest.warns(UserWarning, match=r"WARNING: To avoid overffiting,*"):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize("delta", [np.arange(0, 1, 0.01), (.1, .2), [.4, .5]])
def test_error_delta_wrong_type(delta: Any) -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = MapieMultiLabelClassifier(random_state=42)
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be a float*"):
        mapie_clf.predict(X_toy, method="rcps", delta=delta)


def test_error_partial_fit_different_size() -> None:
    """Test error for RCPS method and delta None"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = MapieMultiLabelClassifier(clf)
    mapie_clf.partial_fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Number of features*"):
        mapie_clf.partial_fit(X, y)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_pipeline_compatibility(strategy: str) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    args = STRATEGIES[strategy][0]
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
        }
    )
    y = np.array(
        [
            [0, 0, 1], [0, 0, 1],
            [1, 1, 0], [1, 0, 1],
            [1, 0, 1], [1, 1, 1]
        ]
    )
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
    pipe = make_pipeline(
        preprocessor,
        MultiOutputClassifier(LogisticRegression())
    )
    pipe.fit(X, y)
    mapie = MapieMultiLabelClassifier(estimator=pipe)
    mapie.fit(X, y)
    mapie.predict(X, method=args["method"], bound=args["bound"], delta=.1)


def test_error_no_fit() -> None:
    """Test error for RCPS method and delta None"""
    clf = WrongOutputModel()
    mapie_clf = MapieMultiLabelClassifier(clf)

    with pytest.raises(
        ValueError,
        match=r".*Please provide a classifier with*"
    ):
        mapie_clf.fit(X_toy, y_toy)


def test_error_estimator_none_partial() -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = MapieMultiLabelClassifier()

    with pytest.raises(
        ValueError,
        match=r".*Invalid estimator with partial_fit*"
    ):
        mapie_clf.partial_fit(X_toy, y_toy)


def test_partial_fit_first_time():
    mclf = MapieMultiLabelClassifier()
    assert mclf._check_partial_fit_first_call()


def test_partial_fit_second_time():
    clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    mclf = MapieMultiLabelClassifier(clf)
    mclf.partial_fit(X, y)
    assert not mclf._check_partial_fit_first_call()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_toy_dataset_predictions(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = MapieMultiLabelClassifier(clf)
    mapie_clf.fit(X_toy, y_toy)
    _, y_ps = mapie_clf.predict(
        X_toy,
        alpha=.2,
        method=args["method"],
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(y_ps[:, :, 0], y_toy_mapie[strategy])
