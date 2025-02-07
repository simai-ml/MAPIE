from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Callable, Tuple, cast

import numpy as np
import pytest
from sklearn.base import RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     PredefinedSplit, RepeatedKFold,
                                     ShuffleSplit, TimeSeriesSplit,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

from mapie._typing import NDArray
from mapie.future.calibrators.ccp import (CCPCalibrator, CustomCCP,
                                          GaussianCCP, PolynomialCCP)
from mapie.conformity_scores import (AbsoluteConformityScore,
                                     GammaConformityScore,
                                     ResidualNormalisedScore)
from mapie.conformity_scores import BaseRegressionScore
from mapie.metrics import regression_coverage_score
from mapie.future.split import SplitCPRegressor

random_state = 1
np.random.seed(random_state)

X_toy = np.linspace(0, 10, num=200).reshape(-1, 1)
y_toy = 2*X_toy[:, 0] + (max(X_toy)/10)*np.random.rand(len(X_toy))
z_toy = np.linspace(0, 10, num=len(X_toy)).reshape(-1, 1)

X, y = make_regression(
    n_samples=200, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]


CV = ["prefit", "split"]

PHI = [
    CustomCCP([lambda X: np.ones((len(X), 1))]),
    PolynomialCCP([0, 1]),
    GaussianCCP(5),
]
WIDTHS = {
    "safe": {
        "split": 4.823,
        "prefit": 4.823,
    },
    "unsafe": {
        "split": 3.867,
        "prefit": 3.867,
    },
}

COVERAGES = {
    "safe": {
        "split": 0.98,
        "prefit": 0.98,
    },
    "unsafe": {
        "split": 0.965,
        "prefit": 0.965,
    },
}


# ======== MapieCCPRegressor =========
def test_initialized() -> None:
    """Test that initialization does not crash."""
    SplitCPRegressor(alpha=0.1)


def test_fit_predictor() -> None:
    """Test that fit_predictor raises no errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit_predictor(X_toy, y_toy)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_calibrator(z: Any) -> None:
    """Test that fit_calibrator raises no errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit_predictor(X_toy, y_toy)
    mapie_reg.fit_calibrator(X_toy, y_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit(z: Any) -> None:
    """Test that fit raises no errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit(X_toy, y_toy, calib_kwargs={"z": z})


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_predictor_fit_calibrator_predict(z: Any) -> None:
    """Test that fit-calibrate-predict raises no errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit_predictor(X_toy, y_toy)
    mapie_reg.fit_calibrator(X_toy, y_toy, z=z)
    mapie_reg.predict(X_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_predict(z: Any) -> None:
    """Test that fit-predict raises no errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit(X_toy, y_toy, calib_kwargs={"z": z})
    mapie_reg.predict(X_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_predict_reg(z: Any) -> None:
    """Test that fit-predict raises no errors."""
    mapie_reg = SplitCPRegressor(calibrator=GaussianCCP(reg_param=0.1),
                                 alpha=0.1)
    mapie_reg.fit(X_toy, y_toy, calib_kwargs={"z": z})
    mapie_reg.predict(X_toy, z=z)


def test_not_fitted_predictor_fit_calibrator() -> None:
    """Test that calibrate before fit raises errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie_reg.fit_calibrator(X_toy, y_toy)


def test_calib_not_complete_phi() -> None:
    """Test that a not complete calibrator definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie_reg = SplitCPRegressor(
            alpha=0.1,
            calibrator=CustomCCP([lambda X: (X < 5).astype(int)], bias=False)
        )
        mapie_reg.fit(X_toy, y_toy)


def test_predict_not_complete_phi() -> None:
    """Test that a not complete calibrator definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie_reg = SplitCPRegressor(
            alpha=0.1,
            calibrator=CustomCCP([lambda X: (X < 5).astype(int)], bias=False)
        )
        mapie_reg.fit(X_toy[X_toy[:, 0] < 5], y_toy[X_toy[:, 0] < 5])
        mapie_reg.predict(X_toy)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie_reg.predict(X_toy)


def test_no_calibrate_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    mapie_reg.fit_predictor(X_toy, y_toy)
    with pytest.raises(NotFittedError):
        mapie_reg.predict(X_toy)


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie_reg = SplitCPRegressor(alpha=0.1)
    assert (
        signature(mapie_reg.fit_predictor).parameters["sample_weight"].default
        is None
    )


@pytest.mark.parametrize("predictor", [0, "a", KFold(), ["a", "b"]])
def test_invalid_predictor(
    predictor: Any
) -> None:
    """Test that invalid predictors raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie = SplitCPRegressor(predictor=predictor, alpha=0.1)
        mapie.fit_predictor(X, y)


@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_invalid_prefit_predictor_calibrate(
    predictor: RegressorMixin,
) -> None:
    """Test that non-fitted predictor with prefit cv raise errors when
    calibrate is called"""
    with pytest.raises(NotFittedError):
        mapie = SplitCPRegressor(predictor=predictor, cv="prefit",
                                 alpha=0.1)
        mapie.fit_calibrator(X, y)


@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_invalid_prefit_predictor_fit(
    predictor: RegressorMixin,
) -> None:
    """Test that non-fitted predictor with prefit cv raise errors when fit
    is called."""
    with pytest.raises(NotFittedError):
        mapie = SplitCPRegressor(predictor=predictor, cv="prefit",
                                 alpha=0.1)
        mapie.fit_predictor(X, y)


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = SplitCPRegressor(random_state=random_state, alpha=0.1)
    mapie_reg.fit(X, y)
    assert isinstance(mapie_reg.predictor_, RegressorMixin)
    assert isinstance(mapie_reg.calibrator_, GaussianCCP)
    assert isinstance(mapie_reg.cv, ShuffleSplit)
    assert mapie_reg.alpha == 0.1
    assert isinstance(mapie_reg.conformity_score_, BaseRegressionScore)
    assert isinstance(mapie_reg.random_state, int)


@pytest.mark.parametrize(
    "alpha", ["a", 0, 2, 1.5, -0.3]
)
def test_invalid_alpha(alpha: Any) -> None:
    with pytest.raises(ValueError):
        mapie = SplitCPRegressor(alpha=alpha)
        mapie.fit(X, y)


@pytest.mark.parametrize(
    "calibrator", [1, "some_string"]
)
def test_invalid_phi(calibrator: Any) -> None:
    with pytest.raises(ValueError):
        mapie = SplitCPRegressor(calibrator=calibrator)
        mapie.fit(X, y)


def test_valid_predictor() -> None:
    """Test that valid predictors are not corrupted"""
    mapie_reg = SplitCPRegressor(
        predictor=DummyRegressor(),
        random_state=random_state,
        alpha=0.1,
    )
    mapie_reg.fit_predictor(X_toy, y_toy)
    assert isinstance(mapie_reg.predictor, DummyRegressor)


@pytest.mark.parametrize(
    "cv", [None, ShuffleSplit(n_splits=1),
           PredefinedSplit(
                test_fold=[1]*(len(X_toy)//2) + [-1]*(len(X_toy)-len(X_toy)//2)
           ), "prefit", "split"]
)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_valid_cv(cv: Any, predictor: RegressorMixin) -> None:
    """Test that valid cv raise no errors."""
    predictor.fit(X_toy, y_toy)
    mapie_reg = SplitCPRegressor(predictor, CustomCCP(bias=True), cv=cv,
                                 alpha=0.1, random_state=random_state)
    mapie_reg.fit(X_toy, y_toy)
    mapie_reg.predict(X_toy)


@pytest.mark.parametrize(
    "cv", ["dummy", 0, 1, 1.5] + [  # Cross val splitters
            3, -1, KFold(n_splits=5), LeaveOneOut(),
            RepeatedKFold(n_splits=5, n_repeats=2), ShuffleSplit(n_splits=5),
            TimeSeriesSplit(), LeavePOut(p=2),
            PredefinedSplit(test_fold=[0]*(len(X_toy)//4) + [1]*(len(X_toy)//4)
                            + [-1]*(len(X_toy)-len(X_toy)//2)),
           ]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid agg_functions raise errors."""
    with pytest.raises(ValueError, match="Invalid cv argument."):
        mapie = SplitCPRegressor(cv=cv, alpha=0.1,
                                 random_state=random_state)
        mapie.fit_predictor(X, y)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("alpha", [0.2])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_fit_calibrate_combined_equivalence(
    alpha: Any, dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, calibrator: CCPCalibrator, predictor: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset

    predictor_1 = clone(predictor)
    predictor_2 = clone(predictor)
    if cv == "prefit":
        predictor_1.fit(X, y)
        predictor_2.fit(X, y)

    np.random.seed(random_state)
    mapie_1 = SplitCPRegressor(
        predictor=predictor_1, calibrator=calibrator,
        cv=cv, alpha=alpha, random_state=random_state
    )
    np.random.seed(random_state)
    mapie_2 = SplitCPRegressor(
        predictor=predictor_2, calibrator=calibrator,
        cv=cv, alpha=alpha, random_state=random_state
    )
    mapie_1.fit(X, y, calib_kwargs={"z": z})
    mapie_2.fit_predictor(X, y)
    mapie_2.fit_calibrator(X, y, z=z)
    y_pred_1, y_pis_1 = mapie_1.predict(X, z=z)
    y_pred_2, y_pis_2 = mapie_2.predict(X, z=z)
    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_predict_output_shape_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, calibrator: CCPCalibrator, predictor: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset
    if cv == "prefit":
        predictor.fit(X, y)

    mapie_reg = SplitCPRegressor(
        predictor=predictor, calibrator=calibrator,
        cv=cv, alpha=0.1, random_state=random_state
    )
    mapie_reg.fit(X, y, calib_kwargs={"z": z})
    y_pred, y_pis = mapie_reg.predict(X, z=z)
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, 1)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_predict_output_shape_no_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, calibrator: CCPCalibrator, predictor: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset
    if cv == "prefit":
        predictor.fit(X, y)

    mapie_reg = SplitCPRegressor(
        predictor=predictor, calibrator=calibrator, cv=cv,
        alpha=None, random_state=random_state
    )
    mapie_reg.fit(X, y, calib_kwargs={"z": z})
    y_pred = mapie_reg.predict(X, z=z)
    assert np.array(y_pred).shape == (X.shape[0],)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("template", PHI)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_same_results_prefit_split(
    dataset: Tuple[NDArray, NDArray, NDArray], template: CCPCalibrator,
    predictor: RegressorMixin,
) -> None:
    """
    Test checking that if split and prefit method have exactly
    the same data split, then we have exactly the same results.
    """
    (X, y, z) = dataset
    cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_index, _ = list(cv.split(X))[0]
    test_fold = np.ones(len(X))
    test_fold[train_index] = -1

    pred_cv = PredefinedSplit(test_fold)
    train_index, val_index = list(pred_cv.split(X, y))[0]
    X_train, X_calib = X[train_index], X[val_index]
    y_train, y_calib = y[train_index], y[val_index]
    z_calib = z[val_index]

    calibrator = cast(CCPCalibrator, clone(template))
    calibrator._transform_params(X, y, z)
    calibrator.init_value = calibrator.init_value_
    if isinstance(calibrator, GaussianCCP):
        calibrator.points = (calibrator.points_, calibrator.sigmas_)

    mapie_1 = SplitCPRegressor(
        clone(predictor), clone(calibrator), pred_cv, alpha=0.1,
        random_state=random_state,
    )

    fitted_predictor = clone(predictor).fit(X_train, y_train)
    mapie_2 = SplitCPRegressor(
        fitted_predictor, clone(calibrator), cv="prefit", alpha=0.1,
        random_state=random_state,
    )

    mapie_1.fit(X, y, calib_kwargs={"z": z})
    mapie_2.fit(X_calib, y_calib, calib_kwargs={"z": z_calib})

    y_pred_1, y_pis_1 = mapie_1.predict(X, z=z)
    y_pred_2, y_pis_2 = mapie_2.predict(X, z=z)

    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_results_for_ordered_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray], cv: Any,
    calibrator: CCPCalibrator, predictor: RegressorMixin
) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    (X, y, z) = dataset
    if cv == "prefit":
        predictor.fit(X, y)

    calibrator._transform_params(X)

    mapie_reg_1 = SplitCPRegressor(predictor, clone(calibrator), cv=cv,
                                   alpha=0.05, random_state=random_state)
    mapie_reg_2 = SplitCPRegressor(predictor, clone(calibrator), cv=cv,
                                   alpha=0.1, random_state=random_state)

    mapie_reg_1.fit(X, y, calib_kwargs={"z": z})
    _, y_pis_1 = mapie_reg_1.predict(X, z=z)
    mapie_reg_2.fit(X, y, calib_kwargs={"z": z})
    _, y_pis_2 = mapie_reg_1.predict(X, z=z)

    assert (y_pis_1[:, 0, 0] <= y_pis_2[:, 0, 0]).all()
    assert (y_pis_1[:, 1, 0] >= y_pis_2[:, 1, 0]).all()


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_results_with_constant_sample_weights(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any,
    predictor: RegressorMixin,
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    (X, y, z) = dataset
    if cv == "prefit":
        predictor.fit(X, y)

    calibrator = cast(CCPCalibrator, clone(PHI[0]))
    calibrator._transform_params(X)
    calibrator.init_value = calibrator.init_value_

    n_samples = len(X)
    mapie0 = SplitCPRegressor(predictor, clone(calibrator),
                              cv=cv, alpha=0.1, random_state=random_state)
    mapie1 = SplitCPRegressor(predictor, clone(calibrator),
                              cv=cv, alpha=0.1, random_state=random_state)
    mapie2 = SplitCPRegressor(predictor, clone(calibrator),
                              cv=cv, alpha=0.1, random_state=random_state)

    mapie0.fit(X, y, sample_weight=None, calib_kwargs={"z": z})
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples),
               calib_kwargs={"z": z})
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 3,
               calib_kwargs={"z": z})

    y_pred0, y_pis0 = mapie0.predict(X, z=z)
    y_pred1, y_pis1 = mapie1.predict(X, z=z)
    y_pred2, y_pis2 = mapie2.predict(X, z=z)
    np.testing.assert_allclose(y_pred0, y_pred1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pred0, y_pred2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pis0, y_pis1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pis0, y_pis2, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dataset", [
    (X, y, z), (X_toy, y_toy, z_toy),
    (np.arange(0, 100).reshape(-1, 1), np.arange(0, 100), None)
])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05])
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_prediction_between_low_up(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any,
    calibrator: CCPCalibrator,
    alpha: float,
    predictor: RegressorMixin
) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    (X, y, z) = dataset

    if cv == "prefit":
        predictor.fit(X, y)

    mapie = SplitCPRegressor(predictor=predictor, calibrator=calibrator,
                             cv=cv, alpha=alpha, random_state=random_state)
    mapie.fit(X, y, calib_kwargs={"z": z})

    with warnings.catch_warnings(record=True) as record:
        y_pred, y_pis = mapie.predict(X, z=z)

    # Check if the warning was issued
    warning_issued = any("The predictions are ill-sorted." in str(w.message)
                         for w in record)

    # Perform assertions based on whether the warning was issued
    if not warning_issued:
        assert (y_pred >= y_pis[:, 0, 0]).all()
        assert (y_pred <= y_pis[:, 1, 0]).all()


@pytest.mark.parametrize("predict_mode", [
    "safe", "unsafe"
])
def test_linear_regression_results(predict_mode: str) -> None:
    """
    Test that the CCPCalibrator method in the case of a constant
    calibrator = x -> np.ones(len(x)), on a multivariate linear regression
    problem with fixed random state, is strictly equivalent to the regular
    CP method (base, jacknife and cv)
    """

    mapie = SplitCPRegressor(
        calibrator=clone(PHI[0]),
        cv=ShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state),
        alpha=0.05,
        random_state=random_state
    )
    mapie.fit(X, y)
    _, y_pis = mapie.predict(
        X, unsafe_approximation=bool(predict_mode == "unsafe")
    )
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(
        width_mean, WIDTHS[predict_mode]["split"], rtol=1e-2
    )
    np.testing.assert_allclose(
        coverage, COVERAGES[predict_mode]["split"], rtol=1e-2
    )


@pytest.mark.parametrize("predict_mode", [
    "safe", "unsafe"
])
def test_results_prefit(predict_mode: str) -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    predictor = LinearRegression().fit(X_train, y_train)
    mapie_reg = SplitCPRegressor(
        predictor=predictor, calibrator=clone(PHI[0]), cv="prefit", alpha=0.05,
        random_state=random_state
    )
    mapie_reg.fit(X_calib, y_calib)
    _, y_pis = mapie_reg.predict(
        X, unsafe_approximation=bool(predict_mode == "unsafe")
    )
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(
        width_mean, WIDTHS[predict_mode]["prefit"], rtol=1e-2
    )
    np.testing.assert_allclose(
        coverage, COVERAGES[predict_mode]["prefit"], rtol=1e-2
    )


@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
@pytest.mark.parametrize(
    "conformity_score", [AbsoluteConformityScore(), GammaConformityScore(),
                         ResidualNormalisedScore()]
)
def test_conformity_score(
    cv: Any,
    calibrator: CCPCalibrator,
    predictor: RegressorMixin,
    conformity_score: BaseRegressionScore,
) -> None:
    """Test that any conformity score function with MAPIE raises no error."""

    if cv == "prefit":
        predictor.fit(X, y + 1e3)

    mapie_reg = SplitCPRegressor(
        predictor=predictor,
        calibrator=calibrator,
        cv=cv,
        alpha=0.1,
        conformity_score=conformity_score,
        random_state=random_state,
    )
    mapie_reg.fit(X, y + 1e3, calib_kwargs={"z": z})
    mapie_reg.predict(X, z=z)


def test_fit_parameters_passing() -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting predictors have used 3 iterations
    only during boosting, instead of default value for n_predictors (=100).
    """
    gb = GradientBoostingRegressor(random_state=random_state)

    mapie_reg = SplitCPRegressor(predictor=gb, alpha=0.1,
                                 random_state=random_state)

    def early_stopping_monitor(i, est, locals):
        """Returns True on the 3rd iteration."""
        if i == 2:
            return True
        else:
            return False

    mapie_reg.fit(X, y, fit_kwargs={"monitor": early_stopping_monitor})

    assert cast(RegressorMixin, mapie_reg.predictor).estimators_.shape[0] == 3


@pytest.mark.parametrize("custom_method", [
    lambda local_arg: local_arg,
    lambda self_arg: self_arg,
    lambda kwarg_arg: kwarg_arg,
    lambda local_arg, *args, **kwargs: local_arg,
    lambda self_arg, *args, **kwargs: self_arg,
    lambda kwarg_arg, *args, **kwargs: kwarg_arg,
])
def test_get_method_arguments(custom_method: Callable) -> None:
    mapie = SplitCPRegressor(alpha=0.1)
    mapie.self_arg = 1
    local_vars = {"local_arg": 1}
    kwarg_args = {"kwarg_arg": 1}

    arguments = mapie._get_method_arguments(custom_method, local_vars,
                                            kwarg_args)
    custom_method(**arguments)


@pytest.mark.parametrize("conformity_scores", [
    np.random.rand(200, 1),
    np.random.rand(200),
])
def test_check_conformity_scores(conformity_scores: NDArray) -> None:
    mapie = SplitCPRegressor()
    assert mapie._check_conformity_scores(conformity_scores).shape == (200,)


def test_check_conformity_scores_error() -> None:
    mapie = SplitCPRegressor()
    with pytest.raises(ValueError, match="Invalid conformity scores."):
        mapie._check_conformity_scores(np.random.rand(200, 5))


def test_optim_kwargs():
    mapie = SplitCPRegressor(alpha=0.1)
    with pytest.warns(UserWarning, match="Iteration limit reached"):
        mapie.fit(
            X, y, calib_kwargs={"method": "SLSQP", "options": {"maxiter": 2}}
        )
