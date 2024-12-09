from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GroupKFold, KFold, LeaveOneOut, PredefinedSplit, ShuffleSplit,
    train_test_split
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.aggregation_functions import aggregate_all
from mapie.conformity_scores import (
    AbsoluteConformityScore, BaseRegressionScore, GammaConformityScore,
    ResidualNormalisedScore
)
from mapie.estimator.regressor import EnsembleRegressor
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=1
)
k = np.ones(shape=(5, X.shape[1]))
METHODS = ["naive", "base", "plus", "minmax"]

random_state = 1


class CustomGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X, check_predict_params=False):
        if check_predict_params:
            return np.zeros(X.shape[0])
        return super().predict(X)


def early_stopping_monitor(i, est, locals):
    """Returns True on the 3rd iteration."""
    if i == 2:
        return True
    else:
        return False


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
    "cv_plus_median": Params(
        method="plus",
        agg_function="median",
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
    "naive": 3.80,
    "split": 3.89,
    "jackknife": 3.89,
    "jackknife_plus": 3.90,
    "jackknife_minmax": 3.96,
    "cv": 3.88,
    "cv_plus": 3.91,
    "cv_minmax": 4.07,
    "prefit": 3.89,
    "cv_plus_median": 3.91,
    "jackknife_plus_ab": 3.90,
    "jackknife_minmax_ab": 4.14,
    "jackknife_plus_median_ab": 3.88,
}

COVERAGES = {
    "naive": 0.954,
    "split": 0.956,
    "jackknife": 0.956,
    "jackknife_plus": 0.952,
    "jackknife_minmax": 0.962,
    "cv": 0.954,
    "cv_plus": 0.954,
    "cv_minmax": 0.962,
    "prefit": 0.956,
    "cv_plus_median": 0.954,
    "jackknife_plus_ab": 0.952,
    "jackknife_minmax_ab": 0.968,
    "jackknife_plus_median_ab": 0.952,
}


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieRegressor()
    assert mapie_reg.agg_function == "mean"
    assert mapie_reg.method == "plus"


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie_reg = MapieRegressor(
        estimator=DummyRegressor(), **STRATEGIES[strategy]
    )
    mapie_reg.fit(X_toy, y_toy)
    assert isinstance(mapie_reg.estimator_.single_estimator_, DummyRegressor)
    for estimator in mapie_reg.estimator_.estimators_:
        assert isinstance(estimator, DummyRegressor)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_reg = MapieRegressor(method=method)
    mapie_reg.fit(X_toy, y_toy)
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)


@pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_agg_function(agg_function: Any) -> None:
    """Test that invalid agg_functions raise errors."""
    mapie_reg = MapieRegressor(agg_function=agg_function)
    with pytest.raises(ValueError, match=r".*Invalid aggregation function.*"):
        mapie_reg.fit(X_toy, y_toy)

    mapie_reg = MapieRegressor(agg_function=None)
    with pytest.raises(
        ValueError,
        match=r".*aggregation function has to be in ['median', 'mean']*"
    ):
        mapie_reg.fit(X_toy, y_toy)
        mapie_reg.predict(X_toy, ensemble=True)


@pytest.mark.parametrize("agg_function", [None, "mean", "median"])
def test_valid_agg_function(agg_function: str) -> None:
    """Test that valid agg_functions raise no errors."""
    mapie_reg = MapieRegressor(agg_function=agg_function)
    mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "cv", [None, -1, 2, KFold(), LeaveOneOut(),
           ShuffleSplit(n_splits=1, test_size=0.5),
           PredefinedSplit(test_fold=[-1]*3+[0]*3),
           "prefit", "split"]
)
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    model = LinearRegression()
    model.fit(X_toy, y_toy)
    mapie_reg = MapieRegressor(estimator=model, cv=cv, test_size=0.5)
    mapie_reg.fit(X_toy, y_toy)
    mapie_reg.predict(X_toy, alpha=0.5)


@pytest.mark.parametrize("cv", [100, 200, 300])
def test_too_large_cv(cv: Any) -> None:
    """Test that too large cv raise sklearn errors."""
    mapie_reg = MapieRegressor(cv=cv)
    with pytest.raises(
        ValueError,
        match=rf".*Cannot have number of splits n_splits={cv} greater.*",
    ):
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.4], (0.2, 0.4)])
def test_predict_output_shape(
    strategy: str, alpha: Any, dataset: Tuple[NDArray, NDArray]
) -> None:
    """Test predict output shape."""
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, n_alpha)


@pytest.mark.parametrize("delta", [0.6, 0.8])
@pytest.mark.parametrize("n_calib", [10 + i for i in range(13)] + [50, 100])
def test_coverage_validity(delta: float, n_calib: int) -> None:
    """
    Test that the prefit method provides valid coverage
    for different calibration data sizes and coverage targets.
    """
    n_split, n_train, n_test = 100, 100, 1000
    n_all = n_train + n_calib + n_test
    X, y = make_regression(n_all, random_state=random_state)
    Xtr, Xct, ytr, yct = train_test_split(
        X, y, train_size=n_train, random_state=random_state
    )

    model = LinearRegression()
    model.fit(Xtr, ytr)

    cov_list = []
    for _ in range(n_split):
        mapie_reg = MapieRegressor(estimator=model, method="base", cv="prefit")
        Xc, Xt, yc, yt = train_test_split(Xct, yct, test_size=n_test)
        mapie_reg.fit(Xc, yc)
        _, y_pis = mapie_reg.predict(Xt, alpha=1-delta)
        y_low, y_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
        coverage = regression_coverage_score(yt, y_low, y_up)
        cov_list.append(coverage)

    # Here we are testing whether the average coverage is statistically
    # less than the target coverage.
    mean_low, mean_up = delta, delta + 1/(n_calib+1)
    _, pval_low = ttest_1samp(cov_list, popmean=mean_low, alternative='less')
    _, pval_up = ttest_1samp(cov_list, popmean=mean_up, alternative='greater')

    # We perform a FWER controlling procedure (Bonferroni)
    p_fwer = 0.01  # probability of making one or more false discoveries: 1%
    p_bonf = p_fwer / 30  # because a total of 30 test_coverage_validity
    np.testing.assert_array_less(p_bonf, pval_low)
    np.testing.assert_array_less(p_bonf, pval_up)


@pytest.mark.parametrize("delta", [0.6, 0.8, 0.9, 0.95])
def test_calibration_data_size_symmetric_score(delta: float) -> None:
    """
    This test function verifies that a ValueError is raised when the number
    of calibration data is lower than the minimum required for the given alpha
    when the conformity score is symmetric. The minimum is calculated as
    1/alpha or 1/(1-delta).
    """
    # Generate data
    n_train, n_all = 100, 1000
    X, y = make_regression(n_all, random_state=42)
    Xtr, Xct, ytr, yct = train_test_split(X, y, train_size=n_train)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(Xtr, ytr)

    # Define a symmetric conformity score
    score = AbsoluteConformityScore(sym=True)

    # Test when the conformity score is symmetric
    # and the number of calibration data is sufficient
    n_calib_sufficient = int(np.ceil(1/(1-delta)))
    Xc, Xt, yc, _ = train_test_split(Xct, yct, train_size=n_calib_sufficient)
    mapie_reg = MapieRegressor(
        estimator=model, method="base", cv="prefit", conformity_score=score
    )
    mapie_reg.fit(Xc, yc)
    mapie_reg.predict(Xt, alpha=1-delta)

    # Test when the conformity score is symmetric
    # and the number of calibration data is insufficient
    with pytest.raises(
        ValueError, match=r"Number of samples of the score is too low*"
    ):
        n_calib_low = int(np.floor(1/(1-delta)))
        Xc, Xt, yc, _ = train_test_split(Xct, yct, train_size=n_calib_low)
        mapie_reg = MapieRegressor(
            estimator=model, method="base", cv="prefit", conformity_score=score
        )
        mapie_reg.fit(Xc, yc)
        mapie_reg.predict(Xt, alpha=1-delta)


@pytest.mark.parametrize("delta", [0.6, 0.8, 0.9, 0.95])
def test_calibration_data_size_asymmetric_score(delta: float) -> None:
    """
    This test function verifies that a ValueError is raised when the number
    of calibration data is lower than the minimum required for the given alpha
    when the conformity score is asymmetric. The minimum is calculated as
    1/alpha or 1/(1-delta).
    """
    # Generate data
    n_train, n_all = 100, 1000
    X, y = make_regression(n_all, random_state=42)
    Xtr, Xct, ytr, yct = train_test_split(X, y, train_size=n_train)

    # Train a model
    model = LinearRegression()
    model.fit(Xtr, ytr)

    # Define an asymmetric conformity score
    score = AbsoluteConformityScore(sym=False)

    # Test when BaseRegressionScore is asymmetric
    # and calibration data size is sufficient
    n_calib_sufficient = int(np.ceil(1/(1-delta) * 2)) + 1
    Xc, Xt, yc, _ = train_test_split(Xct, yct, train_size=n_calib_sufficient)
    mapie_reg = MapieRegressor(
        estimator=model, method="base", cv="prefit", conformity_score=score
    )
    mapie_reg.fit(Xc, yc)
    mapie_reg.predict(Xt, alpha=1-delta)

    # Test when BaseRegressionScore is asymmetric
    # and calibration data size is too low
    with pytest.raises(
        ValueError, match=r"Number of samples of the score is too low*"
    ):
        n_calib_low = int(np.floor(1/(1-delta) * 2))
        Xc, Xt, yc, _ = train_test_split(Xct, yct, train_size=n_calib_low)
        mapie_reg = MapieRegressor(
            estimator=model, method="base", cv="prefit", conformity_score=score
        )
        mapie_reg.fit(Xc, yc)
        mapie_reg.predict(Xt, alpha=1-delta)


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

    mapie_reg = MapieRegressor(method='base', cv=cv)
    mapie_reg.fit(X, y)
    y_pred_1, y_pis_1 = mapie_reg.predict(X, alpha=0.1)

    model = LinearRegression().fit(X_train, y_train)
    mapie_reg = MapieRegressor(estimator=model, method='base', cv="prefit")
    mapie_reg.fit(X_calib, y_calib)
    y_pred_2, y_pis_2 = mapie_reg.predict(X, alpha=0.1)

    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=[0.1, 0.1])
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
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    y_pred_float1, y_pis_float1 = mapie_reg.predict(X, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie_reg.predict(X, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie_reg.predict(X, alpha=alpha)
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
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=[0.05, 0.1])
    assert (y_pis[:, 0, 0] <= y_pis[:, 0, 1]).all()
    assert (y_pis[:, 1, 0] >= y_pis[:, 1, 1]).all()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = MapieRegressor(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = MapieRegressor(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_pis_single = mapie_single.predict(X_toy, alpha=0.5)
    y_pred_multi, y_pis_multi = mapie_multi.predict(X_toy, alpha=0.5)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_pis_single, y_pis_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieRegressor(**STRATEGIES[strategy])
    mapie1 = MapieRegressor(**STRATEGIES[strategy])
    mapie2 = MapieRegressor(**STRATEGIES[strategy])
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


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_groups(strategy: str) -> None:
    """
    Test predictions when groups are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieRegressor(**STRATEGIES[strategy])
    mapie1 = MapieRegressor(**STRATEGIES[strategy])
    mapie2 = MapieRegressor(**STRATEGIES[strategy])
    mapie0.fit(X, y, groups=None)
    mapie1.fit(X, y, groups=np.ones(shape=n_samples))
    mapie2.fit(X, y, groups=np.ones(shape=n_samples) * 5)
    y_pred0, y_pis0 = mapie0.predict(X, alpha=0.05)
    y_pred1, y_pis1 = mapie1.predict(X, alpha=0.05)
    y_pred2, y_pis2 = mapie2.predict(X, alpha=0.05)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


def test_results_with_groups() -> None:
    """
    Test predictions when groups specified (not None and
    not constant).
    """
    X = np.array([0, 10, 20, 0, 10, 20]).reshape(-1, 1)
    y = np.array([0, 10, 20, 0, 10, 20])
    groups = np.array([1, 2, 3, 1, 2, 3])
    estimator = DummyRegressor(strategy="mean")

    strategy_no_group = dict(
        estimator=estimator,
        method="plus",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=False),
    )
    strategy_group = dict(
        estimator=estimator,
        method="plus",
        agg_function="mean",
        cv=GroupKFold(n_splits=3),
    )

    mapie0 = MapieRegressor(**strategy_no_group)
    mapie1 = MapieRegressor(**strategy_group)
    mapie0.fit(X, y, groups=None)
    mapie1.fit(X, y, groups=groups)
    # check class member conformity_scores_ (abs(y - y_pred))
    # cv folds with KFold:
    # [(array([2, 3, 4, 5]), array([0, 1])),
    #  (array([0, 1, 4, 5]), array([2, 3])),
    #  (array([0, 1, 2, 3]), array([4, 5]))]
    # cv folds with GroupKFold:
    # [(array([0, 1, 3, 4]), array([2, 5])),
    #  (array([0, 2, 3, 5]), array([1, 4])),
    #  (array([1, 2, 4, 5]), array([0, 3]))]
    y_pred_0 = [12.5, 12.5, 10, 10, 7.5, 7.5]
    y_pred_1 = [15, 10, 5, 15, 10, 5]
    conformity_scores_0 = np.abs(y - y_pred_0)
    conformity_scores_1 = np.abs(y - y_pred_1)
    np.testing.assert_array_equal(mapie0.conformity_scores_,
                                  conformity_scores_0)
    np.testing.assert_array_equal(mapie1.conformity_scores_,
                                  conformity_scores_1)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_prediction_between_low_up(strategy: str) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=0.1)
    assert (y_pred >= y_pis[:, 0, 0]).all()
    assert (y_pred <= y_pis[:, 1, 0]).all()


@pytest.mark.parametrize("method", ["plus", "minmax"])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
def test_prediction_agg_function(method: str, agg_function: str) -> None:
    """
    Test that predictions differ when ensemble is True/False,
    but not prediction intervals.
    """
    mapie = MapieRegressor(method=method, cv=2, agg_function=agg_function)
    mapie.fit(X, y)
    y_pred_1, y_pis_1 = mapie.predict(X, ensemble=True, alpha=0.1)
    y_pred_2, y_pis_2 = mapie.predict(X, ensemble=False, alpha=0.1)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1, y_pred_2)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_data_confidence_interval(strategy: str) -> None:
    """
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X_toy, y_toy)
    y_pred, y_pis = mapie.predict(X_toy, alpha=0.5)
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 1, 0])
    np.testing.assert_allclose(y_pred, y_pis[:, 0, 0])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X, alpha=0.05)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_results_prefit_ignore_method() -> None:
    """Test that method is ignored when ``cv="prefit"``."""
    estimator = LinearRegression().fit(X, y)
    all_y_pis: List[NDArray] = []
    for method in METHODS:
        mapie_reg = MapieRegressor(
            estimator=estimator, cv="prefit", method=method
        )
        mapie_reg.fit(X, y)
        _, y_pis = mapie_reg.predict(X, alpha=0.1)
        all_y_pis.append(y_pis)
    for y_pis1, y_pis2 in combinations(all_y_pis, 2):
        np.testing.assert_allclose(y_pis1, y_pis2)


def test_results_prefit_naive() -> None:
    """
    Test that prefit, fit and predict on the same dataset
    is equivalent to the "naive" method.
    """
    estimator = LinearRegression().fit(X, y)
    mapie_reg = MapieRegressor(estimator=estimator, method="base", cv="prefit")
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(y, y_pis[:, 0, 0], y_pis[:, 1, 0])
    np.testing.assert_allclose(width_mean, WIDTHS["naive"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["naive"], rtol=1e-2)


def test_results_prefit() -> None:
    """Test prefit results on a standard train/calibration split."""
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=1/2, random_state=1
    )
    estimator = LinearRegression().fit(X_train, y_train)
    mapie_reg = MapieRegressor(estimator=estimator, method="base", cv="prefit")
    mapie_reg.fit(X_calib, y_calib)
    _, y_pis = mapie_reg.predict(X_calib, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(
        y_calib, y_pis[:, 0, 0], y_pis[:, 1, 0]
    )
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


def test_not_enough_resamplings() -> None:
    """
    Test that a warning is raised if at least one conformity score is nan.
    """
    with pytest.warns(UserWarning, match=r"WARNING: at least one point of*"):
        mapie_reg = MapieRegressor(
            cv=Subsample(n_resamplings=2, random_state=0), agg_function="mean"
        )
        mapie_reg.fit(X, y)


def test_no_agg_fx_specified_with_subsample() -> None:
    """
    Test that a warning is raised if at least one conformity score is nan.
    """
    with pytest.raises(
        ValueError, match=r"You need to specify an aggregation*"
    ):
        mapie_reg = MapieRegressor(
            cv=Subsample(n_resamplings=1), agg_function=None
        )
        mapie_reg.fit(X, y)


def test_invalid_aggregate_all() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        aggregate_all(None, X)


def test_aggregate_with_mask_with_prefit() -> None:
    """
    Test ``_aggregate_with_mask`` in case ``cv`` is ``"prefit"``.
    """
    mapie_reg = MapieRegressor(LinearRegression().fit(X, y), cv="prefit")
    mapie_reg = mapie_reg.fit(X, y)
    with pytest.raises(
        ValueError,
        match=r".*There should not be aggregation of predictions.*",
    ):
        mapie_reg.estimator_._aggregate_with_mask(k, k)


def test_aggregate_with_mask_with_invalid_agg_function() -> None:
    """Test ``_aggregate_with_mask`` in case ``agg_function`` is invalid."""
    ens_reg = EnsembleRegressor(
        LinearRegression(),
        "plus",
        KFold(n_splits=5, random_state=None, shuffle=True),
        "nonsense",
        None,
        0.20,
        False
    )
    ens_reg.use_split_method_ = False
    with pytest.raises(
        ValueError,
        match=r".*The value of the aggregation function is not correct*",
    ):
        ens_reg._aggregate_with_mask(k, k)


def test_pred_loof_isnan() -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_reg = MapieRegressor()
    y_pred: NDArray
    mapie_reg = mapie_reg.fit(X, y)
    y_pred, _ = mapie_reg.estimator_._predict_oof_estimator(
        estimator=LinearRegression(),
        X=X_toy,
        val_index=[],
    )
    assert len(y_pred) == 0


def test_pipeline_compatibility() -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
            "y": [5, 7, 3, 9, 10, 8],
        }
    )
    y = pd.Series([5, 7, 3, 9, 10, 8])
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
    pipe = make_pipeline(preprocessor, LinearRegression())
    mapie = MapieRegressor(pipe)
    mapie.fit(X, y)
    mapie.predict(X)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "conformity_score", [AbsoluteConformityScore(), GammaConformityScore()]
)
def test_conformity_score(
    strategy: str, conformity_score: BaseRegressionScore
) -> None:
    """Test that any conformity score function with MAPIE raises no error."""
    mapie_reg = MapieRegressor(
        conformity_score=conformity_score,
        **STRATEGIES[strategy]
    )
    mapie_reg.fit(X, y + 1e3)
    mapie_reg.predict(X, alpha=0.05)


@pytest.mark.parametrize(
    "conformity_score", [ResidualNormalisedScore()]
)
def test_conformity_score_with_split_strategies(
   conformity_score: BaseRegressionScore
) -> None:
    """
    Test that any conformity score function that handle only split strategies
    with MAPIE raises no error.
    """
    mapie_reg = MapieRegressor(
        conformity_score=conformity_score,
        **STRATEGIES["split"]
    )
    mapie_reg.fit(X, y + 1e3)
    mapie_reg.predict(X, alpha=0.05)


@pytest.mark.parametrize("ensemble", [True, False])
def test_return_only_ypred(ensemble: bool) -> None:
    """Test that if return_multi_pred is False it only returns y_pred."""
    mapie_reg = MapieRegressor()
    mapie_reg.fit(X_toy, y_toy)
    output = mapie_reg.estimator_.predict(
        X_toy, ensemble=ensemble, return_multi_pred=False
    )
    assert len(output) == len(X_toy)


@pytest.mark.parametrize("ensemble", [True, False])
def test_return_multi_pred(ensemble: bool) -> None:
    """
    Test that if return_multi_pred is True it returns y_pred and multi_pred.
    """
    mapie_reg = MapieRegressor()
    mapie_reg.fit(X_toy, y_toy)
    output = mapie_reg.estimator_.predict(
        X_toy, ensemble=ensemble, return_multi_pred=True
    )
    assert len(output) == 3


def test_fit_parameters_passing() -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting estimators have used 3 iterations
    only during boosting, instead of default value for n_estimators (=100).
    """
    gb = GradientBoostingRegressor(random_state=random_state)
    mapie = MapieRegressor(estimator=gb, random_state=random_state)
    mapie.fit(X, y, fit_params={'monitor': early_stopping_monitor})

    assert mapie.estimator_.single_estimator_.estimators_.shape[0] == 3
    for estimator in mapie.estimator_.estimators_:
        assert estimator.estimators_.shape[0] == 3


def test_predict_parameters_passing() -> None:
    """
    Test passing predict parameters.
    Checks that y_pred from train are 0, y_pred from test are 0.
    """
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=random_state)
    )
    custom_gbr = CustomGradientBoostingRegressor(random_state=random_state)
    score = AbsoluteConformityScore(sym=True)
    mapie_model = MapieRegressor(estimator=custom_gbr, conformity_score=score)
    predict_params = {'check_predict_params': True}
    mapie_model = mapie_model.fit(
        X_train, y_train, predict_params=predict_params
    )
    y_pred = mapie_model.predict(X_test, **predict_params)
    np.testing.assert_allclose(mapie_model.conformity_scores_, np.abs(y_train))
    np.testing.assert_allclose(y_pred, 0)


def test_fit_params_expected_behavior_unaffected_by_predict_params() -> None:
    """
    We want to verify that there are no interferences
    with predict_params on the expected behavior of fit_params
    Checks that underlying GradientBoosting
    estimators have used 3 iterations only during boosting,
    instead of default value for n_estimators (=100).
    """
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=random_state)
    )
    custom_gbr = CustomGradientBoostingRegressor(random_state=random_state)
    mapie_model = MapieRegressor(estimator=custom_gbr)
    fit_params = {'monitor': early_stopping_monitor}
    predict_params = {'check_predict_params': True}
    mapie_model = mapie_model.fit(
        X_train, y_train,
        fit_params=fit_params, predict_params=predict_params
    )

    assert mapie_model.estimator_.single_estimator_.estimators_.shape[0] == 3
    for estimator in mapie_model.estimator_.estimators_:
        assert estimator.estimators_.shape[0] == 3


def test_predict_params_expected_behavior_unaffected_by_fit_params() -> None:
    """
    We want to verify that there are no interferences
    with fit_params on the expected behavior of predict_params
    Checks that the predictions on the training and test sets
    are 0 for the model with predict_params and that this is not
    the case for the model without predict_params
    """
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=random_state)
    )
    custom_gbr = CustomGradientBoostingRegressor(random_state=random_state)
    score = AbsoluteConformityScore(sym=True)
    mapie_model = MapieRegressor(estimator=custom_gbr, conformity_score=score)
    fit_params = {'monitor': early_stopping_monitor}
    predict_params = {'check_predict_params': True}
    mapie_model = mapie_model.fit(
        X_train, y_train,
        fit_params=fit_params,
        predict_params=predict_params
    )
    y_pred = mapie_model.predict(X_test, **predict_params)

    np.testing.assert_array_equal(mapie_model.conformity_scores_,
                                  np.abs(y_train))
    np.testing.assert_allclose(y_pred, 0)


def test_using_one_predict_parameter_into_predict_but_not_in_fit() -> None:
    """
    Test that using predict parameters in the predict method
    without using predict_parameter in the fit method raises an error.
    """
    custom_gbr = CustomGradientBoostingRegressor(random_state=random_state)
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=random_state)
    )
    mapie = MapieRegressor(estimator=custom_gbr)
    predict_params = {'check_predict_params': True}
    mapie_fitted = mapie.fit(X_train, y_train)

    with pytest.raises(ValueError, match=(
        fr".*Using 'predict_params' '{predict_params}' "
        r"without using one 'predict_params' in the fit method\..*"
        r"Please ensure a similar configuration of 'predict_params' "
        r"is used in the fit method before calling it in predict\..*"
    )):
        mapie_fitted.predict(X_test, **predict_params)


def test_using_one_predict_parameter_into_fit_but_not_in_predict() -> None:
    """
    Test that using predict parameters in the fit method
    without using predict_parameter in
    the predict method raises an error.
    """
    custom_gbr = CustomGradientBoostingRegressor(random_state=random_state)
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2, random_state=random_state)
    )
    mapie = MapieRegressor(estimator=custom_gbr)
    predict_params = {'check_predict_params': True}
    mapie_fitted = mapie.fit(X_train, y_train, predict_params=predict_params)

    with pytest.raises(ValueError, match=(
        r"Using one 'predict_params' in the fit method "
        r"without using one 'predict_params' in the predict method. "
        r"Please ensure a similar configuration of 'predict_params' "
        r"is used in the predict method as called in the fit."
    )):
        mapie_fitted.predict(X_test)


def test_predict_infinite_intervals() -> None:
    """
    Test that MapieRegressor produces infinite bounds with alpha=0
    """
    mapie_reg = MapieRegressor().fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=0., allow_infinite_bounds=True)
    np.testing.assert_allclose(y_pis[:, 0, 0], -np.inf)
    np.testing.assert_allclose(y_pis[:, 1, 0], np.inf)


@pytest.mark.parametrize("method", ["minmax", "naive", "plus", "base"])
@pytest.mark.parametrize("cv", ["split", "prefit"])
def test_check_change_method_to_base(method: str, cv: str) -> None:
    """
    Test of the overloading of method attribute to `base` method in fit
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=random_state
    )
    estimator = LinearRegression().fit(X_train, y_train)
    mapie_reg = MapieRegressor(
        cv=cv, method=method, estimator=estimator
    )
    mapie_reg.fit(X_val, y_val)
    assert mapie_reg.method == "base"


def test_deprecated_ensemble_regressor_fit_warning() -> None:
    ens_reg = EnsembleRegressor(
        LinearRegression(),
        "plus",
        KFold(n_splits=5, random_state=None, shuffle=True),
        "nonsense",
        None,
        0.20,
        False
    )
    with pytest.warns(
        FutureWarning,
        match=r".WARNING: EnsembleRegressor.fit is deprecated.*"
    ):
        ens_reg.fit(X, y)


def test_mapie_regressor_sklearn_estim() -> None:
    """Test that MapieRegressor is an sklearn estimator"""
    check_estimator(MapieRegressor())
