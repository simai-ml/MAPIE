from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mapie_v1.regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    ConformalizedQuantileRegressor
)
from mapiev0.regression import MapieRegressor as MapieRegressorV0 # noqa
from mapiev0.regression import MapieQuantileRegressor as MapieQuantileRegressorV0 # noqa
from mapie_v1.conformity_scores.utils import \
    check_and_select_split_conformity_score
from mapie_v1.integration_tests.utils import (filter_params,
                                              train_test_split_shuffle)
from sklearn.model_selection import KFold

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
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
        "conformity_score": check_and_select_split_conformity_score(
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
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE
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


def initialize_models(
    strategy_key,
    v0_params: dict,
    v1_params: dict,
    k_folds=5,
    random_state=42
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

    elif strategy_key == "cv":
        v0_params.update({"cv": KFold(n_splits=k_folds,
                                      shuffle=True,
                                      random_state=random_state)})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(CrossConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = CrossConformalRegressor(cv=k_folds, **v1_params)

    elif strategy_key == "jackknife":
        v0_params.update({"cv": -1})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(JackknifeAfterBootstrapRegressor.__init__,
                                  v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = JackknifeAfterBootstrapRegressor(**v1_params)

    elif strategy_key == "CQR":
        v0_params = filter_params(MapieQuantileRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieQuantileRegressorV0(**v0_params)
        v1 = ConformalizedQuantileRegressor(**v1_params)

    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    return v0, v1
