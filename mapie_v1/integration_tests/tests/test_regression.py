from __future__ import annotations

import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mapie.conformity_scores import GammaConformityScore, \
    AbsoluteConformityScore
from mapie_v1.regression import SplitConformalRegressor, \
    CrossConformalRegressor

from mapiev0.regression import MapieRegressor as MapieRegressorV0  # noqa

from mapie_v1.conformity_scores._utils import \
    check_and_select_regression_conformity_score
from mapie_v1.integration_tests.utils import (filter_params,
                                              train_test_split_shuffle)
from sklearn.model_selection import LeaveOneOut, GroupKFold

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30

X, y = make_regression(n_samples=500,
                       n_features=10,
                       noise=1.0,
                       random_state=RANDOM_STATE)


@pytest.mark.parametrize("strategy_key", ["split", "prefit"])
@pytest.mark.parametrize("method", ["base", "plus", "minmax"])
@pytest.mark.parametrize("conformity_score", ["absolute"])
@pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("allow_infinite_bounds", [True, False])
@pytest.mark.parametrize(
    "estimator", [
        LinearRegression(),
        RandomForestRegressor(random_state=RANDOM_STATE, max_depth=2)])
@pytest.mark.parametrize("test_size", [0.2, 0.5])
def test_exact_interval_equality_split(
    strategy_key,
    method,
    conformity_score,
    confidence_level,
    agg_function,
    allow_infinite_bounds,
    estimator,
    test_size
):
    """
    Test that the prediction intervals are exactly the same
    between v0 and v1 models when using the same settings.
    """
    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    if strategy_key == "prefit":
        estimator.fit(X_train, y_train)

    v0_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": check_and_select_regression_conformity_score(
            conformity_score
        ),
        "alpha": 1 - confidence_level,
        "agg_function": agg_function,
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "allow_infinite_bounds": allow_infinite_bounds
    }
    v1_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "aggregate_function": agg_function,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
        "allow_infinite_bounds": allow_infinite_bounds
    }

    v0, v1 = initialize_models(
        strategy_key=strategy_key,
        v0_params=v0_params,
        v1_params=v1_params,
    )

    if strategy_key == 'prefit':
        v0.fit(X_conf, y_conf)
    else:
        v0.fit(X, y)
        v1.fit(X_train, y_train)

    v1.conformalize(X_conf, y_conf)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    _, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    np.testing.assert_array_equal(
        v1_pred_intervals,
        v0_pred_intervals,
        err_msg="Prediction intervals differ between v0 and v1 models"
    )


X_cross, y_cross_signed = make_regression(
    n_samples=50,
    n_features=10,
    noise=1.0,
    random_state=RANDOM_STATE
)
y_cross = np.abs(y_cross_signed)
sample_weight = RandomState(RANDOM_STATE).random(len(X_cross))
groups = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
positive_predictor = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=lambda y_: np.log(y_ + 1),
    inverse_func=lambda X_: np.exp(X_) - 1
)

params_test_cases_cross = [
    {
        "v0": {
            "alpha": 0.2,
            "conformity_score": AbsoluteConformityScore(),
            "cv": 4,
            "agg_function": "median",
            "ensemble": True,
            "method": "base",
            "sample_weight": sample_weight,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "cv": 4,
            "aggregation_method": "median",
            "method": "base",
            "fit_params": {"sample_weight": sample_weight},
        }
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "alpha": 0.5,
            "conformity_score": GammaConformityScore(),
            "cv": LeaveOneOut(),
            "method": "plus",
            "optimize_beta": True,
        },
        "v1": {
            "estimator": positive_predictor,
            "confidence_level": 0.5,
            "conformity_score": "gamma",
            "cv": LeaveOneOut(),
            "method": "plus",
            "minimize_interval_width": True,
        }
    },
    {
        "v0": {
            "alpha": 0.1,
            "cv": GroupKFold(),
            "groups": groups,
            "method": "minmax",
            "allow_infinite_bounds": True,
        },
        "v1": {
            "cv": GroupKFold(),
            "groups": groups,
            "method": "minmax",
            "allow_infinite_bounds": True,
        }
    },
]


@pytest.mark.parametrize("params_cross", params_test_cases_cross)
def test_intervals_and_predictions_exact_equality_cross(params_cross):
    v0_params = params_cross["v0"]
    v1_params = params_cross["v1"]

    v0 = MapieRegressorV0(
        **filter_params(MapieRegressorV0.__init__, v0_params)
    )
    v1 = CrossConformalRegressor(
        **filter_params(CrossConformalRegressor.__init__, v1_params)
    )

    v0_fit_params = filter_params(v0.fit, v0_params)
    v1_fit_params = filter_params(v1.fit, v1_params)
    v1_conformalize_params = filter_params(v1.conformalize, v1_params)

    v0.fit(X_cross, y_cross, **v0_fit_params)
    v1.fit(X_cross, y_cross, **v1_fit_params)
    v1.conformalize(X_cross, y_cross, **v1_conformalize_params)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    v1_predict_set_params = filter_params(v1.predict_set, v1_params)

    v0_preds, v0_pred_intervals = v0.predict(X_cross, **v0_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]
    v1_pred_intervals = v1.predict_set(X_cross, **v1_predict_set_params)
    v1_preds = v1.predict(X_cross, **v1_predict_params)

    assert np.equal(v0_preds, v1_preds)
    assert np.equal(v0_pred_intervals, v1_pred_intervals)


def initialize_models(
    strategy_key,
    v0_params: dict,
    v1_params: dict,
):
    if strategy_key == "prefit":
        v0_params.update({"cv": "prefit"})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = SplitConformalRegressor(prefit=True, **v1_params)

    elif strategy_key == "split":
        v0_params.update({"cv": "split"})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = SplitConformalRegressor(**v1_params)

    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    return v0, v1
