from __future__ import annotations
from typing import Optional, Union, Dict, Tuple

import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mapie.subsample import Subsample
from mapie._typing import ArrayLike
from mapie.conformity_scores import GammaConformityScore, \
    AbsoluteConformityScore
from mapie_v1.regression import SplitConformalRegressor, \
    CrossConformalRegressor, \
    JackknifeAfterBootstrapRegressor

from mapiev0.regression import MapieRegressor as MapieRegressorV0  # noqa
from mapie_v1.conformity_scores._utils import \
    check_and_select_regression_conformity_score
from mapie_v1.integration_tests.utils import (filter_params,
                                              train_test_split_shuffle)
from sklearn.model_selection import LeaveOneOut, GroupKFold

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30


X, y_signed = make_regression(
    n_samples=50,
    n_features=10,
    noise=1.0,
    random_state=RANDOM_STATE
)
y = np.abs(y_signed)
sample_weight = RandomState(RANDOM_STATE).random(len(X))
groups = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
positive_predictor = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=lambda y_: np.log(y_ + 1),
    inverse_func=lambda X_: np.exp(X_) - 1
)

X_split, y_split = make_regression(
    n_samples=500,
    n_features=10,
    noise=1.0,
    random_state=RANDOM_STATE
)


@pytest.mark.parametrize("cv", ["split", "prefit"])
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
def test_intervals_and_predictions_exact_equality_split(
    cv,
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
    prefit = cv == "prefit"

    v0_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": check_and_select_regression_conformity_score(
            conformity_score
        ),
        "alpha": 1 - confidence_level,
        "agg_function": agg_function,
        "test_size": test_size,
        "allow_infinite_bounds": allow_infinite_bounds,
        "cv": cv,
        "random_state": RANDOM_STATE,
    }
    v1_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "aggregate_function": agg_function,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
        "allow_infinite_bounds": allow_infinite_bounds,
        "prefit": prefit,
        "random_state": RANDOM_STATE,
    }

    v0, v1 = initialize_models(cv, v0_params, v1_params)
    compare_model_predictions_and_intervals(v0=v0,
                                            v1=v1,
                                            X=X_split,
                                            y=y_split,
                                            v0_params=v0_params,
                                            v1_params=v1_params,
                                            test_size=test_size,
                                            random_state=RANDOM_STATE,
                                            prefit=prefit)


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
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "cv": 4,
            "aggregation_method": "median",
            "method": "base",
            "fit_params": {"sample_weight": sample_weight},
            "random_state": RANDOM_STATE,
        }
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "alpha": [0.5, 0.5],
            "conformity_score": GammaConformityScore(),
            "cv": LeaveOneOut(),
            "method": "plus",
            "optimize_beta": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": positive_predictor,
            "confidence_level": [0.5, 0.5],
            "conformity_score": "gamma",
            "cv": LeaveOneOut(),
            "method": "plus",
            "minimize_interval_width": True,
            "random_state": RANDOM_STATE,
        }
    },
    {
        "v0": {
            "alpha": 0.1,
            "cv": GroupKFold(),
            "groups": groups,
            "method": "minmax",
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "cv": GroupKFold(),
            "groups": groups,
            "method": "minmax",
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        }
    },
]


@pytest.mark.parametrize("params_cross", params_test_cases_cross)
def test_intervals_and_predictions_exact_equality_cross(params_cross):
    v0_params = params_cross["v0"]
    v1_params = params_cross["v1"]

    v0, v1 = initialize_models("cross", v0_params, v1_params)
    compare_model_predictions_and_intervals(v0, v1, X, y, v0_params, v1_params)


params_test_cases_jackknife = [
    {
        "v0": {
            "alpha": 0.2,
            "conformity_score": AbsoluteConformityScore(),
            "cv": Subsample(n_resamplings=10, random_state=RANDOM_STATE),
            "agg_function": "median",
            "ensemble": True,
            "method": "plus",
            "sample_weight": sample_weight,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "resampling": 10,
            "aggregation_method": "median",
            "method": "plus",
            "fit_params": {"sample_weight": sample_weight},
            "random_state": RANDOM_STATE,
        },
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "alpha": [0.5, 0.5],
            "conformity_score": GammaConformityScore(),
            "aggregation_method": "mean",
            "cv": Subsample(n_resamplings=3,
                            replace=True,
                            random_state=RANDOM_STATE),
            "method": "plus",
            "optimize_beta": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": positive_predictor,
            "confidence_level": [0.5, 0.5],
            "aggregation_method": "mean",
            "conformity_score": "gamma",
            "resampling": Subsample(n_resamplings=3,
                                    replace=True,
                                    random_state=RANDOM_STATE),
            "method": "plus",
            "minimize_interval_width": True,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "v0": {
            "alpha": 0.1,
            "cv": Subsample(n_resamplings=10, random_state=RANDOM_STATE),
            "method": "minmax",
            "aggregation_method": "mean",
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.9,
            "resampling": 10,
            "method": "minmax",
            "aggregation_method": "mean",
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        }
    },
]


@pytest.mark.parametrize("params_jackknife", params_test_cases_jackknife)
def test_intervals_and_predictions_exact_equality_jackknife(params_jackknife):
    v0_params = params_jackknife["v0"]
    v1_params = params_jackknife["v1"]

    v0, v1 = initialize_models("jackknife", v0_params, v1_params)
    compare_model_predictions_and_intervals(v0, v1, X, y, v0_params, v1_params)


def initialize_models(
    strategy_key: str,
    v0_params: Dict,
    v1_params: Dict,
) -> Tuple[MapieRegressorV0, Union[
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor
]]:

    v1: Union[SplitConformalRegressor,
              CrossConformalRegressor,
              JackknifeAfterBootstrapRegressor]

    if strategy_key in ["split", "prefit"]:
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v1 = SplitConformalRegressor(**v1_params)

    elif strategy_key == "cross":
        v1_params = filter_params(CrossConformalRegressor.__init__, v1_params)
        v1 = CrossConformalRegressor(**v1_params)

    elif strategy_key == "jackknife":
        v1_params = filter_params(
            JackknifeAfterBootstrapRegressor.__init__,
            v1_params
        )
        v1 = JackknifeAfterBootstrapRegressor(**v1_params)

    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
    v0 = MapieRegressorV0(**v0_params)

    return v0, v1


def compare_model_predictions_and_intervals(
    v0: MapieRegressorV0,
    v1: Union[SplitConformalRegressor,
              CrossConformalRegressor,
              JackknifeAfterBootstrapRegressor],
    X: ArrayLike,
    y: ArrayLike,
    v0_params: Dict = {},
    v1_params: Dict = {},
    prefit: bool = False,
    test_size: Optional[float] = None,
    random_state: int = 42,
) -> None:

    if test_size is not None:
        X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_conf, y_train, y_conf = X, X, y, y

    v0_fit_params = filter_params(v0.fit, v0_params)
    v1_fit_params = filter_params(v1.fit, v1_params)
    v1_conformalize_params = filter_params(v1.conformalize, v1_params)

    if prefit:
        estimator = v0.estimator
        estimator.fit(X_train, y_train)
        v0.estimator = estimator
        v1._mapie_regressor.estimator = estimator

        v0.fit(X_conf, y_conf, **v0_fit_params)
    else:
        v0.fit(X, y, **v0_fit_params)
        v1.fit(X_train, y_train, **v1_fit_params)

    v1.conformalize(X_conf, y_conf, **v1_conformalize_params)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    v1_predict_set_params = filter_params(v1.predict_set, v1_params)

    v0_preds, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_set_params)
    if v1_pred_intervals.ndim == 2:
        v1_pred_intervals = np.expand_dims(v1_pred_intervals, axis=2)

    v1_preds: ArrayLike = v1.predict(X_conf, **v1_predict_params)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_intervals, v1_pred_intervals)
