from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.utils.estimator_checks import check_estimator
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.aggregation_functions import aggregate_all
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

random_state = 1
X_toy = np.array(range(5)).reshape(-1, 1)
y_toy = (5.0 + 2.0 * X_toy ** 1.1).flatten()
X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
k = np.ones(shape=(5, X.shape[1]))
METHODS = ["enbpi"]
UPDATE_DATA = ([6], 17.5)
CONFORMITY_SCORES = [14.189 - 14.038, 17.5 - 18.665]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "agg_function": str,
        "cv": Optional[Union[int, KFold, BlockBootstrap]],
    },
)
STRATEGIES = {
    "jackknife_enbpi_mean_ab_wopt": Params(
        method="enbpi",
        agg_function="mean",
        cv=BlockBootstrap(
            n_resamplings=30, n_blocks=5, random_state=random_state
        ),
    ),
    "jackknife_enbpi_median_ab_wopt": Params(
        method="enbpi",
        agg_function="median",
        cv=BlockBootstrap(
            n_resamplings=30,
            n_blocks=5,
            random_state=random_state,
        ),
    ),
    "jackknife_enbpi_mean_ab": Params(
        method="enbpi",
        agg_function="mean",
        cv=BlockBootstrap(
            n_resamplings=30, n_blocks=5, random_state=random_state
        ),
    ),
    "jackknife_enbpi_median_ab": Params(
        method="enbpi",
        agg_function="median",
        cv=BlockBootstrap(
            n_resamplings=30,
            n_blocks=5,
            random_state=random_state,
        ),
    ),
}

WIDTHS = {
    "jackknife_enbpi_mean_ab_wopt": 3.76,
    "jackknife_enbpi_median_ab_wopt": 3.76,
    "jackknife_enbpi_mean_ab": 3.76,
    "jackknife_enbpi_median_ab": 3.76,
    "prefit": 4.79,

}

COVERAGES = {
    "jackknife_enbpi_mean_ab_wopt": 0.952,
    "jackknife_enbpi_median_ab_wopt": 0.946,
    "jackknife_enbpi_mean_ab": 0.952,
    "jackknife_enbpi_median_ab": 0.946,
    "prefit": 0.98,

}


def test_sklearn_checks() -> None:
    """
    Test that all sklearn estimator checks pass as intended.
    The usage of ``partial_fit`` does not match the sklearn convention
    in the strictest sense since ``partial_fit`` can only be invoked
    after ``fit``; the corresponding estimator check is marked as an
    expected failure.
    """
    check_estimator(MapieTimeSeriesRegressor())


@pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_agg_function(agg_function: Any) -> None:
    """Test that invalid agg_functions raise errors."""

    mapie_ts_reg = MapieTimeSeriesRegressor(agg_function=None)
    with pytest.raises(ValueError, match=r".*If ensemble is True*"):
        mapie_ts_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.4], (0.2, 0.4)])
def test_predict_output_shape(
    strategy: str, alpha: Any, dataset: Tuple[NDArray, NDArray]
) -> None:
    """Test predict output shape."""
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_ts_reg.fit(X, y)
    y_pred, y_pis = mapie_ts_reg.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts_reg.fit(X, y)
    _, y_pis = mapie_ts_reg.predict(X, alpha=[0.1, 0.1])
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts_reg.fit(X, y)
    y_pred_float1, y_pis_float1 = mapie_ts_reg.predict(X, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie_ts_reg.predict(X, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie_ts_reg.predict(X, alpha=alpha)
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_pis_float1[:, :, 0], y_pis_array[:, :, 0])
    np.testing.assert_allclose(y_pis_float2[:, :, 0], y_pis_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_ordered_alpha(strategy: str) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    mapie = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=[0.05, 0.1])
    assert np.all(
        np.abs(y_pis[:, 1, 0] - y_pis[:, 0, 0])
        >= np.abs(y_pis[:, 1, 1] - y_pis[:, 0, 1])
    )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieTimeSeriesRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = MapieTimeSeriesRegressor(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = MapieTimeSeriesRegressor(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_pis_single = mapie_single.predict(X_toy, alpha=0.2)
    y_pred_multi, y_pis_multi = mapie_multi.predict(X_toy, alpha=0.2)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_pis_single, y_pis_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie1 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie2 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
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


@pytest.mark.parametrize("method", ["enbpi"])
@pytest.mark.parametrize("cv", [-1, 2, 3, 5])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_prediction_agg_function(
    method: str, cv: Union[LeaveOneOut, KFold], agg_function: str, alpha: int
) -> None:
    """
    Test that PIs are the same but predictions differ when ensemble is
    True or False.
    """
    mapie = MapieTimeSeriesRegressor(
        method=method, cv=cv, agg_function=agg_function
    )
    mapie.fit(X, y)
    y_pred_1, y_pis_1 = mapie.predict(X, ensemble=True, alpha=alpha)
    y_pred_2, y_pis_2 = mapie.predict(X, ensemble=False, alpha=alpha)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1, y_pred_2)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """

    mapie_ts = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts.fit(X, y)
    if "opt" in strategy:
        optimize_beta = True
    else:
        optimize_beta = False
    _, y_pis = mapie_ts.predict(X, alpha=0.05, optimize_beta=optimize_beta)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()

    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_results_prefit() -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 / 10, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1 / 9, random_state=random_state
    )
    estimator = LinearRegression().fit(X_train, y_train)
    mapie_ts_reg = MapieTimeSeriesRegressor(estimator=estimator, cv="prefit")
    mapie_ts_reg.fit(X_val, y_val)
    _, y_pis = mapie_ts_reg.predict(X_test, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(
        y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
    )
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


def test_not_enough_resamplings() -> None:
    """Test that a warning is raised if at least one residual is nan."""
    with pytest.warns(
        UserWarning,
        match=r"WARNING: at least one point of*"
    ):
        mapie_ts_reg = MapieTimeSeriesRegressor(
            cv=BlockBootstrap(n_resamplings=1, n_blocks=1), agg_function="mean"
        )
        mapie_ts_reg.fit(X, y)


def test_no_agg_fx_specified_with_subsample() -> None:
    """
    Test that an error is raised if ``cv`` is ``BlockBootstrap`` but
    ``agg_function`` is ``None``.
    """
    with pytest.raises(
        ValueError, match=r"You need to specify an aggregation*"
    ):
        mapie_ts_reg = MapieTimeSeriesRegressor(
            cv=BlockBootstrap(n_resamplings=1, n_blocks=1),
            agg_function=None,
        )
        mapie_ts_reg.fit(X, y)


def test_invalid_aggregate_all() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        aggregate_all(None, X)


def test_pred_loof_isnan() -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_ts_reg = MapieTimeSeriesRegressor()
    _, y_pred, _ = mapie_ts_reg._fit_and_predict_oof_model(
        estimator=mapie_ts_reg,
        X=X_toy,
        y=y_toy,
        train_index=[0, 1, 2, 3, 4],
        val_index=[],
    )
    assert len(y_pred) == 0


def test_MapieTimeSeriesRegressor_if_alpha_is_None() -> None:
    """Test ``predict`` when ``alpha`` is None."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1).fit(X_toy, y_toy)

    with pytest.raises(ValueError, match=r".*too many values to unpackt*"):
        y_pred, y_pis = mapie_ts_reg.predict(X_toy, alpha=None)


def test_MapieTimeSeriesRegressor_partial_fit_ensemble() -> None:
    """Test ``partial_fit``."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1).fit(X_toy, y_toy)
    assert round(mapie_ts_reg.conformity_scores_[-1], 2) == round(
        np.abs(CONFORMITY_SCORES[0]), 2
    )
    mapie_ts_reg = mapie_ts_reg.partial_fit(
        X=np.array([UPDATE_DATA[0]]), y=np.array([UPDATE_DATA[1]])
    )
    assert round(mapie_ts_reg.conformity_scores_[-1], 2) == round(
        CONFORMITY_SCORES[1], 2
    )


def test_MapieTimeSeriesRegressor_partial_fit_too_big() -> None:
    """Test ``partial_fit`` raised error."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1).fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*The number of observations*"):
        mapie_ts_reg = mapie_ts_reg.partial_fit(X=X, y=y)


def test_MapieTimeSeriesRegressor_beta_optimize_eeror() -> None:
    """Test ``beta_optimize`` raised error."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1)
    with pytest.raises(ValueError, match=r".*Lower and upper bounds arrays*"):
        mapie_ts_reg._beta_optimize(
            alpha=0.1, upper_bounds=X, lower_bounds=X_toy
        )


def test_deprecated_path_warning() -> None:
    """
    Test that a warning is raised if import with deprecated path.
    """
    with pytest.warns(
        FutureWarning,
        match=r".*WARNING: Deprecated path*"
    ):
        from mapie.time_series_regression import MapieTimeSeriesRegressor
        _ = MapieTimeSeriesRegressor()


def test_consistent_class() -> None:
    """
    Test that importing a class with a new or obsolete path
    produces the same results.
    """
    from mapie.regression import MapieTimeSeriesRegressor as C2
    from mapie.time_series_regression import MapieTimeSeriesRegressor as C1

    mapie_c1 = C1(random_state=random_state).fit(X, y)
    mapie_c2 = C2(random_state=random_state).fit(X, y)

    y_pred_1, y_pis_1 = mapie_c1.predict(X, alpha=0.1)
    y_pred_2, y_pis_2 = mapie_c2.predict(X, alpha=0.1)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    np.testing.assert_allclose(y_pred_1, y_pred_2)
