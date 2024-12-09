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

from mapie._typing import ArrayLike, NDArray
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
    if isinstance(mapie_estimator, MapieClassifier):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )
    if isinstance(mapie_estimator, MapieRegressor):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )


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
    with pytest.raises(
        ValueError, match="(Invalid method.)|(Invalid conformity score.)*"
    ):
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


def test_warning_when_import_from_gamma_conformity_score():
    """Check that a DepreciationWarning is raised when importing from
    mapie.conformity_scores.residual_conformity_scores"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.conformity_scores.residual_conformity_scores import (
            GammaConformityScore
        )
        GammaConformityScore()


def test_warning_when_import_from_absolute_conformity_score():
    """Check that a DepreciationWarning is raised when importing from
    mapie.conformity_scores.residual_conformity_scores"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.conformity_scores.residual_conformity_scores import (
            AbsoluteConformityScore
        )
        AbsoluteConformityScore()


def test_warning_when_import_from_residual_conformity_score():
    """Check that a DepreciationWarning is raised when importing from
    mapie.conformity_scores.residual_conformity_scores"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.conformity_scores.residual_conformity_scores import (
            ResidualNormalisedScore
        )
        ResidualNormalisedScore()


def test_warning_when_import_from_conformity_scores():
    """Check that a DepreciationWarning is raised when importing from
    mapie.conformity_scores.conformity_score"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.conformity_scores.conformity_scores import (
            ConformityScore
        )

        class DummyConformityScore(ConformityScore):
            def __init__(self) -> None:
                super().__init__(sym=True, consistency_check=True)

            def get_signed_conformity_scores(
                self, y: ArrayLike, y_pred: ArrayLike, **kwargs
            ) -> NDArray:
                return np.array([])

            def get_estimation_distribution(
                self, y_pred: ArrayLike, conformity_scores: ArrayLike, **kwargs
            ) -> NDArray:
                """
                A positive constant is added to the sum between predictions and
                conformity scores to make the estimated distribution
                inconsistent with the conformity score.
                """
                return np.array([])

            def get_conformity_scores(
                self, y: ArrayLike, y_pred: ArrayLike, **kwargs
            ) -> NDArray:
                return np.array([])

            def predict_set(
                self, X: NDArray, alpha_np: NDArray, **kwargs
            ) -> NDArray:
                return np.array([])

        dcs = DummyConformityScore()
        dcs.get_signed_conformity_scores(y_toy, y_toy)
        dcs.get_estimation_distribution(y_toy, y_toy)
        dcs.get_conformity_scores(y_toy, y_toy)
        dcs.predict_set(y_toy, 0.5)


def test_warning_when_import_from_old_get_true_label_position():
    """Check that a DepreciationWarning is raised when importing from
    mapie.conformity_scores.residual_conformity_scores"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.conformity_scores.utils_classification_conformity_scores\
              import get_true_label_position
        get_true_label_position(np.array([[0.1, 0.2, 0.7]]), np.array([2]))


def test_warning_when_import_from_estimator():
    """Check that a DepreciationWarning is raised when importing from
    mapie.estimator.estimator"""

    with pytest.warns(
        FutureWarning, match=r".*WARNING: Deprecated path to import.*"
    ):
        from mapie.estimator.estimator import EnsembleRegressor
        EnsembleRegressor(
            estimator=LinearRegression(),
            method="naive",
            cv=3,
            agg_function="mean",
            n_jobs=1,
            test_size=0.2,
            verbose=0,
        )
