from __future__ import annotations

from inspect import signature
from typing import Any, Optional

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split
from typing_extensions import TypedDict

from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor

X_toy = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
     5, 0, 1, 2, 3, 4, 5]
).reshape(-1, 1)
y_toy = np.array(
    [5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9,
     11, 13, 15, 5, 7, 9, 11, 13, 15]
)

random_state = 1

qt = QuantileRegressor(solver="highs-ds")
gb = GradientBoostingRegressor(
    loss="quantile",
    random_state=random_state
)

X, y = make_regression(
    n_samples=500,
    n_features=10,
    noise=1.0,
    random_state=random_state
)

X_train, X_calib, y_train, y_calib = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=random_state
)

ESTIMATOR = [qt, gb]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "alpha": float,
        "random_state": Optional[int],
    },
)

STRATEGIES = {
    "quantile_alpha2":
        Params(method="plus", alpha=0.2, random_state=random_state),
    "quantile_alpha3":
        Params(method="plus", alpha=0.3, random_state=random_state),
    "quantile_alpha4":
        Params(method="plus", alpha=0.4, random_state=random_state),
    "quantile_alpha8":
        Params(method="plus", alpha=0.8, random_state=random_state),
}

WIDTHS = {
    "quantile_alpha2": 2.744639410027496,
    "quantile_alpha3": 2.100088918692794,
    "quantile_alpha4": 1.7101104296999252,
    "quantile_alpha8": 0.4907343583658844
}


COVERAGES = {
    "quantile_alpha2": 0.84,
    "quantile_alpha3": 0.732,
    "quantile_alpha4": 0.608,
    "quantile_alpha8": 0.206
}


class NotFitPredictEstimator:
    def __init__(self, alpha):
        self.alpha = alpha


class NoLossPamameterEstimator(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict(self, *args: Any) -> None:
        """Dummy predict."""


class NoAlphaPamameterEstimator(BaseEstimator):
    def __init__(self, alpha, loss):
        self.alpha = alpha
        self.loss = loss

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict(self, *args: Any) -> None:
        """Dummy predict."""


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieQuantileRegressor()
    assert mapie_reg.method == "plus"
    assert mapie_reg.cv is None
    assert mapie_reg.alpha == 0.1


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie_reg = MapieQuantileRegressor()
    assert (
        signature(mapie_reg.fit).parameters["sample_weight"].default
        is None
    )


def test_default_parameters_estimator() -> None:
    """Test default values of estimator."""
    mapie_reg = MapieQuantileRegressor()
    mapie_reg.fit(X, y)
    for estimator in mapie_reg.single_estimator_alpha_:
        assert isinstance(estimator, QuantileRegressor)
        assert estimator.__dict__["solver"] == "highs-ds"


def test_no_predict_fit_estimator() -> None:
    """Test that estimators with not fit or predict methods raise an error."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid estimator.*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=NotFitPredictEstimator(alpha=0.2)
        )
        mapie_reg.fit(X_toy, y_toy)


def test_no_para_loss_estimator() -> None:
    """Test to check when it does not have a valid loss_name."""
    with pytest.raises(
        ValueError,
        match=r".*The matching parameter `loss_name`*",
    ):
        mapie_reg = MapieQuantileRegressor()
        mapie_reg.quantile_estimator_params[
            "NoLossPamameterEstimator"
        ] = {
            "loss_name": "noloss",
            "alpha_name": "alpha"
        }
        mapie_reg.estimator = NoLossPamameterEstimator(alpha=0.2)
        mapie_reg.fit(X_toy, y_toy)


def test_no_para_alpha_estimator() -> None:
    """Test to check when it does not have a valid alpha parameter name"""
    with pytest.raises(
        ValueError,
        match=r".*The matching parameter `alpha_name`*",
    ):
        mapie_reg = MapieQuantileRegressor()
        mapie_reg.quantile_estimator_params[
            "NoAlphaPamameterEstimator"
        ] = {
            "loss_name": "loss",
            "alpha_name": "noalpha"
        }
        mapie_reg.estimator = NoAlphaPamameterEstimator(
            alpha=0.2,
            loss="quantile"
        )
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_results_for_same_alpha(estimator: RegressorMixin) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        alpha=0.2,
        random_state=random_state
    )
    mapie_reg_clone = clone(mapie_reg)
    mapie_reg.fit(X, y)
    mapie_reg_clone.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, alpha=0.2)
    y_pred_clone, y_pis_clone = mapie_reg_clone.predict(X, alpha=0.2)
    np.testing.assert_allclose(y_pred, y_pred_clone)
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis_clone[:, 0, 0])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis_clone[:, 1, 0])


@pytest.mark.parametrize("alphas", ["hello", MapieQuantileRegressor, [2], 1])
def test_wrong_alphas_types(alphas: float) -> None:
    """Checking for wrong type of alphas"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid alpha. Allowed values are float.*",
    ):
        mapie_reg = MapieQuantileRegressor(alpha=alphas)
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("alphas", [1.0, 1.6, 1.95, 5.0, -0.1, -0.001, -10.0])
def test_wrong_alphas(alphas: float) -> None:
    """Checking for alphas values that are too big according to all value."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid alpha. Allowed values are between 0 and .*",
    ):
        mapie_reg = MapieQuantileRegressor(alpha=alphas)
        mapie_reg.fit(X, y)


def test_estimators_quantile_function() -> None:
    """Checking for badly set estimator parameters."""
    with pytest.raises(
        ValueError,
        match=r".*You need to set the loss/objective*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor()
            )
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("est", [RandomForestClassifier(), LinearRegression()])
def test_estimators_not_in_list(est: RegressorMixin) -> None:
    """
    Test for estimators that are not in the list, hence not accepted
    estimators
    """
    with pytest.raises(
        ValueError,
        match=r".*The base model does not seem to be accepted by.*",
    ):
        mapie_reg = MapieQuantileRegressor(estimator=est)
        mapie_reg.fit(X_toy, y_toy)


def test_for_small_dataset() -> None:
    """Test for when we have calibration datasets that are too small."""
    with pytest.raises(
        ValueError,
        match=r".*Number of samples of the score is too low*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=qt,
            alpha=0.1
        )
        mapie_reg.fit(
            np.array([1, 2, 3]),
            np.array([2, 2, 3])
        )


# Working but want to add both symmetry and different estimators
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a different strategies.
    """
    alpha = STRATEGIES[strategy]['alpha']
    mapie = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X, alpha=alpha)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_error_model_prefit() -> None:
    """
    Check that the model is an iterable object for "prefit".
    """
    with pytest.raises(
        ValueError,
        match=r".*Estimator for prefit must be an iterable object*"
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=object(), cv="prefit"
        )
        mapie_reg.fit(X, y)


def test_quantile_prefit_three_estimators() -> None:
    """
    Test that there is a list with three estimators provided for
    cv="prefit".
    """
    with pytest.raises(
        ValueError,
        match=r".*You need to have provided 3 different estimators, th*",
    ):
        gb_trained1, gb_trained2 = clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        list_estimators = [gb_trained1, gb_trained2]
        mapie_reg = MapieQuantileRegressor(
            estimator=list_estimators,
            cv="prefit"
        )
        mapie_reg.fit(X_calib, y_calib)


def test_prefit_no_fit_predict() -> None:
    """
    Check that the estimators given have a prefit and fit attribute.
    """
    with pytest.raises(
        ValueError,
        match=r"Invalid estimator. Please provide a regressor with fit and*",
    ):
        gb_trained1, gb_trained2 = clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        gb_trained3 = 3
        list_estimators = [gb_trained1, gb_trained2, gb_trained3]
        mapie_reg = MapieQuantileRegressor(
            estimator=list_estimators,
            cv="prefit",
            alpha=0.3
        )
        mapie_reg.fit(X_calib, y_calib)


def test_non_trained_estimator() -> None:
    """
    Check that the estimators are all already trained when used in prefit.
    """
    with pytest.raises(
        ValueError,
        match=r".*instance is not fitted yet. Call 'fit' with appropriate*",
    ):
        gb_trained1, gb_trained2, gb_trained3 = clone(gb), clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        list_estimators = [gb_trained1, gb_trained2, gb_trained3]
        mapie_reg = MapieQuantileRegressor(
            estimator=list_estimators,
            cv="prefit",
            alpha=0.3
        )
        mapie_reg.fit(X_calib, y_calib)


def test_warning_alpha_prefit() -> None:
    """
    Check that the user is warned that the alphas need to be correctly set.
    """
    with pytest.warns(
        UserWarning,
        match=r".*WARNING: The alpha that is set needs to be the same*"
    ):
        gb_trained1, gb_trained2, gb_trained3 = clone(gb), clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        gb_trained3.fit(X_train, y_train)
        list_estimators = [gb_trained1, gb_trained2, gb_trained3]
        mapie_reg = MapieQuantileRegressor(
            estimator=list_estimators,
            cv="prefit",
            alpha=0.3
        )
        mapie_reg.fit(X_calib, y_calib)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3])
def test_prefit_and_non_prefit_equal(alpha: float) -> None:
    """
    Check that when using prefit and not prefit, the same values
    are found.
    """
    list_estimators = []
    alphas_ = [alpha/2, 1-(alpha/2), 0.5]
    for alpha_ in alphas_:
        est = clone(qt)
        params = {"quantile": alpha_}
        est.set_params(**params)
        est.fit(X, y)
        list_estimators.append(est)
    mapie_reg_prefit = MapieQuantileRegressor(
        estimator=list_estimators,
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit.fit(X, y)
    y_pred_prefit, y_pis_prefit = mapie_reg_prefit.predict(X, alpha=alpha)

    mapie_reg = MapieQuantileRegressor(estimator=qt, alpha=alpha)
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, alpha=alpha)

    np.testing.assert_allclose(y_pred_prefit, y_pred)
    np.testing.assert_allclose(y_pis_prefit, y_pis)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3])
def test_prefit_different_type_list_tuple_array(alpha: float) -> None:
    """
    Check that the type of Iterable (list, np.array, tuple) to
    estimators gives similar results.
    """
    list_estimators = []
    alphas_ = [alpha/2, 1-(alpha/2), 0.5]
    for alpha_ in alphas_:
        est = clone(qt)
        params = {"quantile": alpha_}
        est.set_params(**params)
        est.fit(X_train, y_train)
        list_estimators.append(est)

    mapie_reg_prefit_list = MapieQuantileRegressor(
        estimator=list_estimators,
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit_list.fit(X_calib, y_calib)
    y_pred_prefit_list, y_pis_prefit_list = \
        mapie_reg_prefit_list.predict(X, alpha=alpha)

    mapie_reg_prefit_tuple = MapieQuantileRegressor(
        estimator=tuple(list_estimators),
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit_tuple.fit(X_calib, y_calib)
    y_pred_prefit_tuple, y_pis_prefit_tuple = \
        mapie_reg_prefit_tuple.predict(X, alpha=alpha)

    mapie_reg_prefit_array = MapieQuantileRegressor(
        estimator=np.array(list_estimators),
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit_array.fit(X_calib, y_calib)
    y_pred_prefit_array, y_pis_prefit_array = \
        mapie_reg_prefit_array.predict(X, alpha=alpha)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_tuple)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_tuple)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_array)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_array)
