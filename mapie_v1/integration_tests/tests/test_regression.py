from __future__ import annotations
from typing import Optional, Union, Dict, Type

import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from mapie.subsample import Subsample
from mapie._typing import ArrayLike
from mapie.conformity_scores import GammaConformityScore, \
    AbsoluteConformityScore, ResidualNormalisedScore
from mapie_v1.regression import SplitConformalRegressor, \
    CrossConformalRegressor, \
    JackknifeAfterBootstrapRegressor, \
    ConformalizedQuantileRegressor

from mapiev0.regression import MapieRegressor as MapieRegressorV0  # noqa
from mapiev0.regression import MapieQuantileRegressor as MapieQuantileRegressorV0  # noqa
from mapie_v1.integration_tests.utils import (filter_params,
                                              train_test_split_shuffle)
from sklearn.model_selection import LeaveOneOut, GroupKFold

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30
N_SAMPLES = 200
N_GROUPS = 5

X, y_signed = make_regression(
    n_samples=N_SAMPLES,
    n_features=10,
    noise=1.0,
    random_state=RANDOM_STATE
)
y = np.abs(y_signed)
sample_weight = RandomState(RANDOM_STATE).random(len(X))
groups = [j for j in range(N_GROUPS) for i in range((N_SAMPLES//N_GROUPS))]
positive_predictor = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=lambda y_: np.log(y_ + 1),
    inverse_func=lambda X_: np.exp(X_) - 1
)

sample_weight_train = train_test_split(
    X,
    y,
    sample_weight,
    test_size=0.4,
    random_state=RANDOM_STATE
)[-2]

params_test_cases_split = [
    {
        "v0": {
            "alpha": 0.2,
            "conformity_score": AbsoluteConformityScore(),
            "cv": "split",
            "test_size": 0.4,
            "sample_weight": sample_weight,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "prefit": False,
            "test_size": 0.4,
            "fit_params": {"sample_weight": sample_weight_train},
        }
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "test_size": 0.2,
            "alpha": [0.5, 0.5],
            "conformity_score": GammaConformityScore(),
            "cv": "split",
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": positive_predictor,
            "test_size": 0.2,
            "confidence_level": [0.5, 0.5],
            "conformity_score": "gamma",
            "prefit": False,
        }
    },
    {
        "v0": {
            "estimator": LinearRegression(),
            "alpha": 0.1,
            "test_size": 0.2,
            "conformity_score": ResidualNormalisedScore(
                random_state=RANDOM_STATE
            ),
            "cv": "prefit",
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": LinearRegression(),
            "confidence_level": 0.9,
            "prefit": True,
            "test_size": 0.2,
            "conformity_score": ResidualNormalisedScore(
                random_state=RANDOM_STATE
            ),
            "allow_infinite_bounds": True,
        }
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "alpha": 0.1,
            "conformity_score": GammaConformityScore(),
            "cv": "split",
            "random_state": RANDOM_STATE,
            "test_size": 0.3,
            "optimize_beta": True
        },
        "v1": {
            "estimator": positive_predictor,
            "confidence_level": 0.9,
            "conformity_score": GammaConformityScore(),
            "test_size": 0.3,
            "minimize_interval_width": True
        }
    },
]


@pytest.mark.parametrize("params_split", params_test_cases_split)
def test_intervals_and_predictions_exact_equality_split(params_split):
    v0_params = params_split["v0"]
    v1_params = params_split["v1"]

    test_size = v1_params.get("test_size", None)
    prefit = v1_params.get("prefit", False)

    compare_model_predictions_and_intervals(
        model_v0=MapieRegressorV0,
        model_v1=SplitConformalRegressor,
        X=X,
        y=y,
        v0_params=v0_params,
        v1_params=v1_params,
        test_size=test_size,
        prefit=prefit,
        random_state=RANDOM_STATE,
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
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "cv": 4,
            "aggregate_predictions": "median",
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

    compare_model_predictions_and_intervals(
        model_v0=MapieRegressorV0,
        model_v1=CrossConformalRegressor,
        X=X,
        y=y,
        v0_params=params_cross["v0"],
        v1_params=params_cross["v1"],
        random_state=RANDOM_STATE,
    )


params_test_cases_jackknife = [
    {
        "v0": {
            "alpha": 0.2,
            "conformity_score": AbsoluteConformityScore(),
            "cv": Subsample(n_resamplings=15, random_state=RANDOM_STATE),
            "agg_function": "median",
            "ensemble": True,
            "method": "plus",
            "sample_weight": sample_weight,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "resampling": Subsample(
                n_resamplings=15, random_state=RANDOM_STATE
            ),
            "aggregation_method": "median",
            "method": "plus",
            "fit_params": {"sample_weight": sample_weight},
            "ensemble": True,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "v0": {
            "estimator": positive_predictor,
            "alpha": [0.5, 0.5],
            "conformity_score": GammaConformityScore(),
            "agg_function": "mean",
            "cv": Subsample(n_resamplings=20,
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
            "resampling": Subsample(n_resamplings=20,
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
            "cv": Subsample(n_resamplings=30, random_state=RANDOM_STATE),
            "method": "minmax",
            "agg_function": "mean",
            "ensemble": True,
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.9,
            "resampling": Subsample(
                n_resamplings=30, random_state=RANDOM_STATE
            ),
            "method": "minmax",
            "aggregation_method": "mean",
            "ensemble": True,
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        }
    },
]


@pytest.mark.parametrize("params_jackknife", params_test_cases_jackknife)
def test_intervals_and_predictions_exact_equality_jackknife(params_jackknife):

    compare_model_predictions_and_intervals(
        model_v0=MapieRegressorV0,
        model_v1=JackknifeAfterBootstrapRegressor,
        X=X,
        y=y,
        v0_params=params_jackknife["v0"],
        v1_params=params_jackknife["v1"],
        random_state=RANDOM_STATE,
    )


split_model = QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )

gbr_models = []
gbr_alpha = 0.1

for alpha_ in [gbr_alpha / 2, (1 - (gbr_alpha / 2)), 0.5]:
    estimator_ = GradientBoostingRegressor(
        loss='quantile',
        alpha=alpha_,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    gbr_models.append(estimator_)

params_test_cases_quantile = [
    {
        "v0": {
            "alpha": 0.2,
            "cv": "split",
            "method": "quantile",
            "calib_size": 0.4,
            "sample_weight": sample_weight,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "confidence_level": 0.8,
            "prefit": False,
            "test_size": 0.4,
            "fit_params": {"sample_weight": sample_weight_train},
        },
    },
    {
        "v0": {
            "estimator": gbr_models,
            "alpha": gbr_alpha,
            "cv": "prefit",
            "method": "quantile",
            "calib_size": 0.2,
            "sample_weight": sample_weight,
            "optimize_beta": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": gbr_models,
            "confidence_level": 1-gbr_alpha,
            "prefit": True,
            "test_size": 0.2,
            "fit_params": {"sample_weight": sample_weight},
            "minimize_interval_width": True,
        },
    },
    {
        "v0": {
            "estimator": split_model,
            "alpha": 0.5,
            "cv": "split",
            "method": "quantile",
            "calib_size": 0.3,
            "allow_infinite_bounds": True,
            "random_state": RANDOM_STATE,
        },
        "v1": {
            "estimator": split_model,
            "confidence_level": 0.5,
            "prefit": False,
            "test_size": 0.3,
            "allow_infinite_bounds": True,
        },
    },
    {
        "v0": {
            "alpha": 0.1,
            "cv": "split",
            "method": "quantile",
            "calib_size": 0.3,
            "random_state": RANDOM_STATE,
            "symmetry": False
        },
        "v1": {
            "confidence_level": 0.9,
            "prefit": False,
            "test_size": 0.3,
            "symmetric_intervals": False,
        },
    },
]


@pytest.mark.parametrize("params_quantile", params_test_cases_quantile)
def test_intervals_and_predictions_exact_equality_quantile(params_quantile):
    v0_params = params_quantile["v0"]
    v1_params = params_quantile["v1"]

    test_size = v1_params.get("test_size", None)
    prefit = v1_params.get("prefit", False)

    compare_model_predictions_and_intervals(
        model_v0=MapieQuantileRegressorV0,
        model_v1=ConformalizedQuantileRegressor,
        X=X,
        y=y,
        v0_params=v0_params,
        v1_params=v1_params,
        test_size=test_size,
        prefit=prefit,
        random_state=RANDOM_STATE,
    )


def compare_model_predictions_and_intervals(
    model_v0: Type[MapieRegressorV0],
    model_v1: Type[Union[
        SplitConformalRegressor,
        CrossConformalRegressor,
        JackknifeAfterBootstrapRegressor,
        ConformalizedQuantileRegressor
    ]],
    X: ArrayLike,
    y: ArrayLike,
    v0_params: Dict = {},
    v1_params: Dict = {},
    prefit: bool = False,
    test_size: Optional[float] = None,
    sample_weight: Optional[ArrayLike] = None,
    random_state: int = RANDOM_STATE,
) -> None:

    if test_size is not None:
        X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        X_train, X_conf, y_train, y_conf = X, X, y, y

    if prefit:
        estimator = v0_params["estimator"]
        if isinstance(estimator, list):
            for single_estimator in estimator:
                single_estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train)

        v0_params["estimator"] = estimator
        v1_params["estimator"] = estimator

    v0_init_params = filter_params(model_v0.__init__, v0_params)
    v1_init_params = filter_params(model_v1.__init__, v1_params)

    v0 = model_v0(**v0_init_params)
    v1 = model_v1(**v1_init_params)

    v0_fit_params = filter_params(v0.fit, v0_params)
    v1_fit_params = filter_params(v1.fit, v1_params)
    v1_conformalize_params = filter_params(v1.conformalize, v1_params)

    if prefit:
        v0.fit(X_conf, y_conf, **v0_fit_params)
    else:
        v0.fit(X, y, **v0_fit_params)
        v1.fit(X_train, y_train, **v1_fit_params)

    v1.conformalize(X_conf, y_conf, **v1_conformalize_params)

    v0_predict_params = filter_params(v0.predict, v0_params)
    if 'alpha' in v0_init_params:
        v0_predict_params.pop('alpha')

    v1_predict_params = filter_params(v1.predict, v1_params)
    v1_predict_set_params = filter_params(v1.predict_set, v1_params)

    v0_preds, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_set_params)
    if v1_pred_intervals.ndim == 2:
        v1_pred_intervals = np.expand_dims(v1_pred_intervals, axis=2)

    v1_preds: ArrayLike = v1.predict(X_conf, **v1_predict_params)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_intervals, v1_pred_intervals)
