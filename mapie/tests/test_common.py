from inspect import signature
from typing import Any, List, Tuple

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from mapie.classification import MapieClassifier
from mapie.regression import MapieQuantileRegressor, MapieRegressor

X_toy = np.arange(18).reshape(-1, 1)
y_toy = np.array(
    [0, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 1, 2, 2]
    )


def MapieSimpleEstimators() -> List[BaseEstimator]:
    return [MapieRegressor, MapieClassifier]


def MapieEstimators() -> List[BaseEstimator]:
    return [MapieRegressor, MapieClassifier, MapieQuantileRegressor]


def MapieDefaultEstimators() -> List[BaseEstimator]:
    return [
        (MapieRegressor, LinearRegression),
        (MapieClassifier, LogisticRegression),
    ]


def MapieTestEstimators() -> List[BaseEstimator]:
    return [
        (MapieRegressor, LinearRegression()),
        (MapieRegressor, make_pipeline(LinearRegression())),
        (MapieClassifier, LogisticRegression()),
        (MapieClassifier, make_pipeline(LogisticRegression())),
    ]


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_initialized(MapieEstimator: BaseEstimator) -> None:
    """Test that initialization does not crash."""
    MapieEstimator()


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_parameters(MapieEstimator: BaseEstimator) -> None:
    """Test default values of input parameters."""
    mapie_estimator = MapieEstimator()
    assert mapie_estimator.estimator is None
    assert mapie_estimator.cv is None
    assert mapie_estimator.verbose == 0
    assert mapie_estimator.n_jobs is None


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit(MapieEstimator: BaseEstimator) -> None:
    """Test that fit raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that fit-predict raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)
    mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_no_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that predict before fit raises errors."""
    mapie_estimator = MapieEstimator()
    with pytest.raises(NotFittedError):
        mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_default_sample_weight(MapieEstimator: BaseEstimator) -> None:
    """Test default sample weights."""
    mapie_estimator = MapieEstimator()
    assert (
        signature(mapie_estimator.fit).parameters["sample_weight"].default
        is None
    )


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_default_alpha(MapieEstimator: BaseEstimator) -> None:
    """Test default alpha."""
    mapie_estimator = MapieEstimator()
    assert (
        signature(mapie_estimator.predict).parameters["alpha"].default is None
    )


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that None estimator defaults to expected value."""
    MapieEstimator, DefaultEstimator = pack
    mapie_estimator = MapieEstimator(estimator=None)
    mapie_estimator.fit(X_toy, y_toy)
    assert isinstance(mapie_estimator.single_estimator_, DefaultEstimator)


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_invalid_estimator(
    MapieEstimator: BaseEstimator, estimator: Any
) -> None:
    """Test that invalid estimators raise errors."""
    mapie_estimator = MapieEstimator(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_invalid_prefit_estimator(
    pack: Tuple[BaseEstimator, BaseEstimator]
) -> None:
    """Test that non-fitted estimator with prefit cv raise errors."""
    MapieEstimator, estimator = pack
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    with pytest.raises(NotFittedError):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_valid_prefit_estimator(
    pack: Tuple[BaseEstimator, BaseEstimator]
) -> None:
    """Test that fitted estimators with prefit cv raise no errors."""
    MapieEstimator, estimator = pack
    estimator.fit(X_toy, y_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    check_is_fitted(mapie_estimator, mapie_estimator.fit_attributes)
    assert mapie_estimator.n_features_in_ == 1


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
@pytest.mark.parametrize("method", [0.5, 1, "cv", ["base", "plus"]])
def test_invalid_method(MapieEstimator: BaseEstimator, method: str) -> None:
    """Test that invalid methods raise errors."""
    mapie_estimator = MapieEstimator(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
@pytest.mark.parametrize(
    "cv", [-3.14, -2, 0, 1, "cv", LinearRegression(), [1, 2]]
)
def test_invalid_cv(MapieEstimator: BaseEstimator, cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie_estimator = MapieEstimator(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_alpha_results(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """
    Test that alpha set to ``None`` in MapieEstimator gives same predictions
    as base estimator.
    """
    MapieEstimator, DefaultEstimator = pack
    estimator = DefaultEstimator()
    estimator.fit(X_toy, y_toy)
    y_pred_expected = estimator.predict(X_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    y_pred = mapie_estimator.predict(X_toy)
    np.testing.assert_allclose(y_pred_expected, y_pred)


@parametrize_with_checks([MapieRegressor()])
def test_sklearn_compatible_estimator(
    estimator: BaseEstimator, check: Any
) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    check(estimator)
