from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.conformity_scores import (AbsoluteConformityScore, ConformityScore,
                                     GammaConformityScore)
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.subsample import Subsample

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

X_toy, y_toy = make_regression(
    n_samples=50, n_features=10, noise=1.0, random_state=1
)
X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=1
)


MAPIE_ESTIMATORS = [
    MapieRegressor,
    MapieQuantileRegressor
]

MAPIE_SINGLE_ESTIMATORS = [
    MapieRegressor
]

BASE_ESTIMATORS = {
    "MapieRegressor": LinearRegression,
    "MapieQuantileRegressor": QuantileRegressor
}

PREFIT_ESTIMATORS = {
    "MapieRegressor": LinearRegression().fit(X, y),
    "MapieQuantileRegressor": [
        QuantileRegressor(quantile=0.05).fit(X, y),
        QuantileRegressor(quantile=0.50).fit(X, y),
        QuantileRegressor(quantile=0.95).fit(X, y)
    ]
}

METHODS = ["naive", "base", "plus", "minmax"]

random_state = 1

Params = TypedDict(
    "Params",
    {
        "method": str,
        "agg_function": str,
        "cv": Optional[Union[int, KFold, Subsample]],
        "test_size": Optional[Union[int, float]],
        "random_state": Optional[int],
    },
)

STRATEGIES = {
    "naive": Params(
        method="naive",
        agg_function="median",
        cv=None,
        test_size=None,
        random_state=random_state
    ),
    "split": Params(
        method="base",
        agg_function="median",
        cv="split",
        test_size=0.5,
        random_state=random_state
    ),
    "jackknife": Params(
        method="base",
        agg_function="mean",
        cv=-1,
        test_size=None,
        random_state=random_state
    ),
    "jackknife_plus": Params(
        method="plus",
        agg_function="mean",
        cv=-1,
        test_size=None,
        random_state=random_state
    ),
    "jackknife_minmax": Params(
        method="minmax",
        agg_function="mean",
        cv=-1,
        test_size=None,
        random_state=random_state
    ),
    "cv": Params(
        method="base",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
    "cv_plus": Params(
        method="plus",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
    "cv_minmax": Params(
        method="minmax",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
    "jackknife_plus_ab": Params(
        method="plus",
        agg_function="mean",
        cv=Subsample(n_resamplings=30, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
    "jackknife_minmax_ab": Params(
        method="minmax",
        agg_function="mean",
        cv=Subsample(n_resamplings=30, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
    "jackknife_plus_median_ab": Params(
        method="plus",
        agg_function="median",
        cv=Subsample(n_resamplings=30, random_state=random_state),
        test_size=None,
        random_state=random_state
    ),
}

WIDTHS = {
    "MapieRegressor": {
        "naive": 3.81,
        "split": 3.87,
        "jackknife": 3.89,
        "jackknife_plus": 3.9,
        "jackknife_minmax": 3.96,
        "cv": 3.84,
        "cv_plus": 3.90,
        "cv_minmax": 4.04,
        "prefit": 4.81,
        "cv_plus_median": 3.9,
        "jackknife_plus_ab": 3.89,
        "jackknife_minmax_ab": 4.14,
        "jackknife_plus_median_ab": 3.88
    },
    "MapieQuantileRegressor": {
        "naive": 3.56,
        "split": 3.94,
        "jackknife": 4.03,
        "jackknife_plus": 3.96,
        "jackknife_minmax": 4.27,
        "cv": 4.29,
        "cv_plus": 4.22,
        "cv_minmax": 4.71,
        "prefit": 4.81,
        "cv_plus_median": 3.9,
        "jackknife_plus_ab": 4.01,
        "jackknife_minmax_ab": 4.56,
        "jackknife_plus_median_ab": 4.1}}
COVERAGES = {
    "MapieRegressor": {
        "naive": 0.952,
        "split": 0.952,
        "jackknife": 0.952,
        "jackknife_plus": 0.952,
        "jackknife_minmax": 0.952,
        "cv": 0.952,
        "cv_plus": 0.954,
        "cv_minmax": 0.962,
        "prefit": 0.98,
        "cv_plus_median": 0.954,
        "jackknife_plus_ab": 0.954,
        "jackknife_minmax_ab": 0.968,
        "jackknife_plus_median_ab": 0.952},
    "MapieQuantileRegressor": {
        "naive": 0.952,
        "split": 0.968,
        "jackknife": 0.97,
        "jackknife_plus": 0.966,
        "jackknife_minmax": 0.974,
        "cv": 0.972,
        "cv_plus": 0.972,
        "cv_minmax": 0.984,
        "prefit": 0.98,
        "cv_plus_median": 0.954,
        "jackknife_plus_ab": 0.962,
        "jackknife_minmax_ab": 0.984,
        "jackknife_plus_median_ab": 0.972
    }
}


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_default_parameters(estimator: BaseEstimator) -> None:
    """Test default values of input parameters."""
    mapie_est = estimator()
    assert mapie_est.agg_function == "mean"
    assert mapie_est.method == "plus"


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(estimator: BaseEstimator, strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    base_estimator = BASE_ESTIMATORS[estimator.__name__]
    mapie_reg = estimator(estimator=base_estimator(), **STRATEGIES[strategy])
    mapie_reg.fit(X_toy, y_toy)
    assert isinstance(mapie_reg.single_estimator_, base_estimator)
    for elt in mapie_reg.estimators_:
        assert isinstance(elt, base_estimator) or \
            list(map(type, elt)).count(base_estimator) == len(elt)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("method", METHODS)
def test_valid_method(estimator: BaseEstimator, method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_reg = estimator(method=method)
    mapie_reg.fit(X_toy, y_toy)
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_agg_function(
    estimator: BaseEstimator, agg_function: Any
) -> None:
    """Test that invalid agg_functions raise errors."""
    mapie_reg = estimator(agg_function=agg_function)
    with pytest.raises(ValueError, match=r".*Invalid aggregation function.*"):
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_invalid_agg_function_as_none(estimator: BaseEstimator) -> None:
    """Test that invalid agg_functions raise errors."""
    mapie_reg = estimator(agg_function=None)
    with pytest.raises(ValueError, match=r".*If ensemble is True*"):
        mapie_reg.fit(X_toy, y_toy)
        mapie_reg.predict(X_toy, ensemble=True)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("agg_function", [None, "mean", "median"])
def test_valid_agg_function(
    estimator: BaseEstimator, agg_function: str
) -> None:
    """Test that valid agg_functions raise no errors."""
    mapie_reg = estimator(agg_function=agg_function)
    mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize(
    "cv", [None, -1, 2, KFold(), LeaveOneOut(),
           ShuffleSplit(n_splits=1), "split", "prefit"]
)
def test_valid_cv(estimator: BaseEstimator, cv: Any) -> None:
    """Test that valid cv raise no errors."""
    if cv != "prefit":
        model = BASE_ESTIMATORS[estimator.__name__]()
    else:
        model = PREFIT_ESTIMATORS[estimator.__name__]
    mapie_reg = estimator(estimator=model, cv=cv)
    mapie_reg.fit(X_toy, y_toy)
    mapie_reg.predict(X_toy, alpha=0.5)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("cv", [100, 200, 300])
def test_too_large_cv(estimator: BaseEstimator, cv: Any) -> None:
    """Test that too large cv raise sklearn errors."""
    mapie_reg = estimator(cv=cv)
    with pytest.raises(
        ValueError,
        match=rf".*Cannot have number of splits n_splits={cv} greater.*",
    ):
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.4], (0.2, 0.4)])
def test_predict_output_shape(
    estimator: BaseEstimator, strategy: str, alpha: Any,
    dataset: Tuple[NDArray, NDArray]
) -> None:
    """Test predict output shape."""
    mapie_reg = estimator(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, n_alpha)


def test_same_results_prefit_split() -> None:
    """
    Test checking that if split and prefit method have exactly
    the same data split, then we have exactly the same results.
    """
    X, y = make_regression(
        n_samples=500, n_features=10, noise=1.0, random_state=1
    )
    cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_index, val_index = list(cv.split(X))[0]
    X_train, X_calib = X[train_index], X[val_index]
    y_train, y_calib = y[train_index], y[val_index]

    mapie_reg = MapieRegressor(cv=cv)
    mapie_reg.fit(X, y)
    y_pred_1, y_pis_1 = mapie_reg.predict(X, alpha=0.1)

    model = LinearRegression().fit(X_train, y_train)
    mapie_reg = MapieRegressor(estimator=model, cv="prefit")
    mapie_reg.fit(X_calib, y_calib)
    y_pred_2, y_pis_2 = mapie_reg.predict(X, alpha=0.1)

    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(
    estimator: BaseEstimator, strategy: str
) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = estimator(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=[0.1, 0.1])
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    estimator: BaseEstimator, strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie_reg = estimator(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    y_pred_float1, y_pis_float1 = mapie_reg.predict(X, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie_reg.predict(X, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie_reg.predict(X, alpha=alpha)
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_pis_float1[:, :, 0], y_pis_array[:, :, 0])
    np.testing.assert_allclose(y_pis_float2[:, :, 0], y_pis_array[:, :, 1])


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_ordered_alpha(
    estimator: BaseEstimator, strategy: str
) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    mapie_reg = estimator(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=[0.05, 0.1])
    assert (y_pis[:, 0, 0] <= y_pis[:, 0, 1]).all()
    assert (y_pis[:, 1, 0] >= y_pis[:, 1, 1]).all()


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(
    estimator: BaseEstimator, strategy: str
) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = estimator(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = estimator(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_pis_single = mapie_single.predict(X_toy, alpha=0.2)
    y_pred_multi, y_pis_multi = mapie_multi.predict(X_toy, alpha=0.2)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_pis_single, y_pis_multi)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(
    estimator: BaseEstimator, strategy: str
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = estimator(**STRATEGIES[strategy])
    mapie1 = estimator(**STRATEGIES[strategy])
    mapie2 = estimator(**STRATEGIES[strategy])
    mapie0.fit(X, y, sample_weight=None)
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_pis0 = mapie0.predict(X, alpha=0.05)
    y_pred1, y_pis1 = mapie1.predict(X, alpha=0.05)
    y_pred2, y_pis2 = mapie2.predict(X, alpha=0.05)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_prediction_between_low_up(
    estimator: BaseEstimator, strategy: str
) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = estimator(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=0.1)
    assert (y_pred >= y_pis[:, 0, 0]).all()
    assert (y_pred <= y_pis[:, 1, 0]).all()


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("method", ["plus", "minmax"])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
def test_prediction_agg_function(
    estimator: BaseEstimator, method: str, agg_function: str
) -> None:
    """
    Test that predictions differ when ensemble is True/False,
    but not prediction intervals.
    """
    mapie = estimator(method=method, cv=2, agg_function=agg_function)
    mapie.fit(X, y)
    y_pred_1, y_pis_1 = mapie.predict(X, ensemble=True, alpha=0.1)
    y_pred_2, y_pis_2 = mapie.predict(X, ensemble=False, alpha=0.1)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1, y_pred_2)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(
    estimator: BaseEstimator, strategy: str
) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """
    mapie = estimator(**STRATEGIES[strategy])
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X, alpha=0.05)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(
        width_mean, WIDTHS[estimator.__name__][strategy], rtol=1e-2
    )
    np.testing.assert_allclose(
        coverage, COVERAGES[estimator.__name__][strategy], rtol=1e-2
    )


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_results_prefit_ignore_method(estimator: BaseEstimator) -> None:
    """Test that method is ignored when ``cv="prefit"``."""
    model = PREFIT_ESTIMATORS[estimator.__name__]
    all_y_pis: List[NDArray] = []
    for method in METHODS:
        mapie_reg = estimator(estimator=model, cv="prefit", method=method)
        mapie_reg.fit(X, y)
        _, y_pis = mapie_reg.predict(X, alpha=0.1)
        all_y_pis.append(y_pis)
    for y_pis1, y_pis2 in combinations(all_y_pis, 2):
        np.testing.assert_allclose(y_pis1, y_pis2)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_not_enough_resamplings(estimator: BaseEstimator) -> None:
    """
    Test that a warning is raised if at least one conformity score is nan.
    """
    with pytest.warns(UserWarning, match=r"WARNING: at least one point of*"):
        mapie_reg = estimator(
            cv=Subsample(n_resamplings=1), agg_function="mean"
        )
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_no_agg_fx_specified_with_subsample(estimator: BaseEstimator) -> None:
    """
    Test that a warning is raised if at least one conformity score is nan.
    """
    with pytest.raises(
        ValueError, match=r"You need to specify an aggregation*"
    ):
        mapie_reg = estimator(
            cv=Subsample(n_resamplings=1), agg_function=None
        )
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_aggregate_with_mask_with_prefit(estimator) -> None:
    """
    Test ``_aggregate_with_mask`` in case ``cv`` is ``"prefit"``.
    """
    k = None
    mapie_reg = estimator(cv="prefit")
    with pytest.raises(
        ValueError,
        match=r".*There should not be aggregation of predictions if cv is*",
    ):
        mapie_reg._aggregate_with_mask(k, k)

    mapie_reg = estimator(agg_function="nonsense")
    with pytest.raises(
        ValueError,
        match=r".*The value of self.agg_function is not correct*",
    ):
        mapie_reg._aggregate_with_mask(k, k)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_pred_oof_isnan(estimator: BaseEstimator) -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_reg = estimator()
    _, y_pred, _ = mapie_reg._fit_and_predict_oof_model(
        estimator=BASE_ESTIMATORS[estimator.__name__](),
        X=X_toy,
        y=y_toy,
        train_index=[0, 1, 2, 3, 4],
        val_index=[],
    )
    assert len(y_pred) == 0


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
def test_pipeline_compatibility(estimator: BaseEstimator) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X_toy, y_toy = make_regression(
        n_samples=100, n_features=1, noise=1.0, random_state=1
    )
    X = pd.DataFrame(
        {
            "x_cat": np.random.choice(
                ["A", "B"], X_toy.shape[0], replace=True
            ),
            "x_num": X_toy.flatten(),
            "y": y_toy,
        }
    )
    y = pd.Series(y_toy)
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[("encoding", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"]),
        ]
    )
    pipe = make_pipeline(preprocessor, BASE_ESTIMATORS[estimator.__name__]())
    mapie = estimator(pipe)
    mapie.fit(X, y)
    mapie.predict(X)


@pytest.mark.parametrize("estimator", MAPIE_ESTIMATORS)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "conformity_score", [AbsoluteConformityScore(), GammaConformityScore()]
)
def test_gammaconformityscore(
    estimator: BaseEstimator, strategy: str, conformity_score: ConformityScore
) -> None:
    """Test that GammaConformityScore with MAPIE raises no error."""
    mapie_reg = estimator(
        conformity_score=conformity_score,
        **STRATEGIES[strategy]
    )
    mapie_reg.fit(X, y + 1e3)
    _, y_pis = mapie_reg.predict(X, alpha=0.05)
