from __future__ import annotations

from inspect import signature
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieQuantileRegressor

X_toy = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
     5, 0, 1, 2, 3, 4, 5]
).reshape(-1, 1)
y_toy = np.array(
    [5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9,
     11, 13, 15, 5, 7, 9, 11, 13, 15]
)

random_state = 1

X_train_toy, X_calib_toy, y_train_toy, y_calib_toy = train_test_split(
    X_toy,
    y_toy,
    test_size=0.5,
    random_state=random_state
)

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

SYMMETRY = [True, False]
ESTIMATOR = [qt, gb]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "alpha": float,
    },
)

STRATEGIES = {
    "quantile_alpha2": Params(method="quantile", alpha=0.2),
    "quantile_alpha3": Params(method="quantile", alpha=0.3),
    "quantile_alpha4": Params(method="quantile", alpha=0.4),
    "quantile_alpha8": Params(method="quantile", alpha=0.8),
    }

WIDTHS = {
    "quantile_alpha2": 2.7360884795455576,
    "quantile_alpha3": 2.185652142101473,
    "quantile_alpha4": 1.731718678152845,
    "quantile_alpha8": 0.66909752420949,
}

COVERAGES = {
    "quantile_alpha2": 0.834,
    "quantile_alpha3": 0.738,
    "quantile_alpha4": 0.646,
    "quantile_alpha8": 0.264,
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
    assert mapie_reg.method == "quantile"
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
    mapie_reg.fit(
        X_train,
        y_train,
        X_calib=X_calib,
        y_calib=y_calib
        )
    for estimator in mapie_reg.estimators_:
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
        mapie_reg.fit(
            X_train_toy,
            y_train_toy,
            X_calib=X_calib_toy,
            y_calib=y_calib_toy
            )


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
        mapie_reg.estimator = NoLossPamameterEstimator(
            alpha=0.2
            )
        mapie_reg.fit(
            X_train_toy,
            y_train_toy,
            X_calib=X_calib_toy,
            y_calib=y_calib_toy
        )


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
        mapie_reg.fit(
            X_train_toy,
            y_train_toy,
            X_calib=X_calib_toy,
            y_calib=y_calib_toy
        )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_valid_method(strategy: str, estimator: RegressorMixin) -> None:
    """Test that valid strategies and estimators raise no error"""
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie_reg.fit(
        X_train_toy,
        y_train_toy,
        X_calib=X_calib_toy,
        y_calib=y_calib_toy
    )
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)
    assert mapie_reg.__dict__["method"] == "quantile"


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("dataset", [
        (X_train, X_calib, y_train, y_calib),
        (X_train_toy, X_calib_toy, y_train_toy, y_calib_toy)
    ]
)
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_predict_output_shape(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
    symmetry: bool
) -> None:
    """Test predict output shape."""
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    (X_t, X_c, y_t, y_c) = dataset
    mapie_reg.fit(X_t, y_t, X_calib=X_c, y_calib=y_c)
    y_pred, y_pis = mapie_reg.predict(X_t, symmetry=symmetry)
    assert y_pred.shape == (X_t.shape[0],)
    assert y_pis[:, 0, 0].shape == (X_t.shape[0],)
    assert y_pis[:, 1, 0].shape == (X_t.shape[0],)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(
    strategy: str,
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X_train)
    mapie0 = MapieQuantileRegressor(
        estimator=qt,
        **STRATEGIES[strategy]
        )
    mapie1 = MapieQuantileRegressor(
        estimator=qt,
        **STRATEGIES[strategy]
        )
    mapie2 = MapieQuantileRegressor(
        estimator=qt,
        **STRATEGIES[strategy]
        )
    mapie0.fit(
        X_train,
        y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        sample_weight=None
        )
    mapie1.fit(
        X_train,
        y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        sample_weight=np.ones(shape=n_samples)
        )
    mapie2.fit(
        X_train,
        y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        sample_weight=np.ones(shape=n_samples) * 5
        )

    np.testing.assert_allclose(
        mapie0.conformity_scores_,
        mapie1.conformity_scores_
    )
    np.testing.assert_allclose(
        mapie0.conformity_scores_,
        mapie2.conformity_scores_
    )

    y_pred0, y_pis0 = mapie0.predict(X)
    y_pred1, y_pis1 = mapie1.predict(X)
    y_pred2, y_pis2 = mapie2.predict(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_results_for_same_alpha(
    estimator: RegressorMixin,
    symmetry: bool
) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        alpha=0.2
    )
    mapie_reg_clone = clone(mapie_reg)
    mapie_reg.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    mapie_reg_clone.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    y_pred, y_pis = mapie_reg.predict(X, symmetry=symmetry)
    y_pred_clone, y_pis_clone = mapie_reg_clone.predict(X, symmetry=symmetry)
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
        mapie_reg.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)


@pytest.mark.parametrize("alphas", [1.0, 1.6, 1.95, 5.0, -0.1, -0.001, -10.0])
def test_wrong_alphas(alphas: float) -> None:
    """Checking for alphas values that are too big according to all value."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid alpha. Allowed values are between .*",
    ):
        mapie_reg = MapieQuantileRegressor(alpha=alphas)
        mapie_reg.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)


def test_estimators_quantile_function() -> None:
    """Checking for badly set estimator parameters."""
    with pytest.raises(
        ValueError,
        match=r".*You need to set the loss/objective*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor()
            )
        mapie_reg.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)


@pytest.mark.parametrize("cv", [-1, 2, KFold(), LeaveOneOut()])
def test_invalid_cv(cv: Any) -> None:
    """Test that valid cv raise errors."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv method.*",
    ):
        mapie = MapieQuantileRegressor(cv=cv)
        mapie.fit(
            X_train_toy,
            y_train_toy,
            X_calib=X_calib_toy,
            y_calib=y_calib_toy
        )


@pytest.mark.parametrize("cv", [None, "split"])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieQuantileRegressor(cv=cv)
    mapie.fit(
        X_train_toy,
        y_train_toy,
        X_calib=X_calib_toy,
        y_calib=y_calib_toy
    )


def test_calib_dataset_is_none() -> None:
    """Test that the fit method works when X_calib or y_calib is None."""
    mapie = MapieQuantileRegressor()
    mapie.fit(X, y, calib_size=0.5)
    mapie.predict(X)


def test_calib_dataset_is_none_with_sample_weight() -> None:
    """
    Test that the fit method works with calib dataset defined is None
    with sample weights.
    """
    mapie = MapieQuantileRegressor()
    mapie.fit(X, y, calib_size=0.5, sample_weight=np.ones(X.shape[0]))
    mapie.predict(X)


def test_calib_dataset_is_none_vs_defined() -> None:
    """
    Test that for the same results whether you split before
    or in the fit method.
    """
    mapie = MapieQuantileRegressor()
    mapie_defined = clone(mapie)
    mapie.fit(X, y, calib_size=0.5, random_state=random_state)
    mapie_defined.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    y_pred, y_pis = mapie.predict(X)
    y_pred_defined, y_pis_defined = mapie_defined.predict(X)
    np.testing.assert_allclose(y_pred, y_pred_defined, rtol=1e-2)
    np.testing.assert_allclose(y_pis, y_pis_defined, rtol=1e-2)


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
        mapie_reg.fit(
            X_train_toy,
            y_train_toy,
            X_calib=X_calib_toy,
            y_calib=y_calib_toy
        )


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
            np.array([2, 2, 3]),
            X_calib=np.array([3, 5]),
            y_calib=np.array([2, 3])
        )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("dataset", [
    (X_train, X_calib, y_train, y_calib),
    (X_train_toy, X_calib_toy, y_train_toy, y_calib_toy)
])
def test_conformity_len(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
) -> None:
    """Test conformity scores output shape."""
    (X_t, X_c, y_t, y_c) = dataset
    n_samples = int(len(X_c))
    mapie_regressor = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie_regressor.fit(X_t, y_t, X_calib=X_c, y_calib=y_c)
    assert mapie_regressor.conformity_scores_[0].shape[0] == n_samples


# Working but want to add both symmetry and different estimators
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a different strategies.
    """
    mapie = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    _, y_pis = mapie.predict(X)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


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
        mapie_reg.fit(
            X_calib,
            y_calib
        )


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
        mapie_reg.fit(
            X_calib,
            y_calib
        )


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
        mapie_reg.fit(
            X_calib,
            y_calib
        )


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
        mapie_reg.fit(
            X_calib,
            y_calib
        )


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
        est.fit(X_train, y_train)
        list_estimators.append(est)
    mapie_reg_prefit = MapieQuantileRegressor(
        estimator=list_estimators,
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit.fit(X_calib, y_calib)
    y_pred_prefit, y_pis_prefit = mapie_reg_prefit.predict(X)

    mapie_reg = MapieQuantileRegressor(
        estimator=qt,
        alpha=alpha
    )
    mapie_reg.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    y_pred, y_pis = mapie_reg.predict(X)

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
    y_pred_prefit_list, y_pis_prefit_list = mapie_reg_prefit_list.predict(X)

    mapie_reg_prefit_tuple = MapieQuantileRegressor(
        estimator=tuple(list_estimators),
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit_tuple.fit(X_calib, y_calib)
    y_pred_prefit_tuple, y_pis_prefit_tuple = mapie_reg_prefit_tuple.predict(X)

    mapie_reg_prefit_array = MapieQuantileRegressor(
        estimator=np.array(list_estimators),
        cv="prefit",
        alpha=alpha
    )
    mapie_reg_prefit_array.fit(X_calib, y_calib)
    y_pred_prefit_array, y_pis_prefit_array = mapie_reg_prefit_array.predict(X)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_tuple)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_tuple)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_array)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_array)


@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_pipeline_compatibility(estimator: RegressorMixin) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B", "A", "B", "B", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5, 4, 3, np.nan, 3],
            "y": [5, 7, 3, 9, 10, 8, 9, 7, 9, 8]
        }
    )
    y = pd.Series([5, 7, 3, 9, 10, 8, 9, 7, 10, 5])
    X_train_toy, X_calib_toy, y_train_toy, y_calib_toy = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=random_state
    )
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"])
        ]
    )
    pipe = make_pipeline(preprocessor, estimator)
    mapie = MapieQuantileRegressor(pipe, alpha=0.4)
    mapie.fit(
        X_train_toy,
        y_train_toy,
        X_calib=X_calib_toy,
        y_calib=y_calib_toy
    )
    mapie.predict(X)


def test_deprecated_path_warning() -> None:
    """
    Test that a warning is raised if import with deprecated path.
    """
    with pytest.warns(
        FutureWarning,
        match=r".*WARNING: Deprecated path*"
    ):
        from mapie.quantile_regression import MapieQuantileRegressor
        _ = MapieQuantileRegressor()


def test_consistent_class() -> None:
    """
    Test that importing a class with a new or obsolete path
    produces the same results.
    """
    from mapie.quantile_regression import MapieQuantileRegressor as C1
    from mapie.regression import MapieQuantileRegressor as C2

    mapie_c1 = C1(alpha=0.1).fit(X, y, random_state=random_state)
    mapie_c2 = C2(alpha=0.1).fit(X, y, random_state=random_state)

    y_pred_1, y_pis_1 = mapie_c1.predict(X, alpha=0.1)
    y_pred_2, y_pis_2 = mapie_c2.predict(X, alpha=0.1)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    np.testing.assert_allclose(y_pred_1, y_pred_2)
