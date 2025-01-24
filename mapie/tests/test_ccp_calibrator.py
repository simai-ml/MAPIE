from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ShuffleSplit

from mapie.future.calibrators.ccp import (CCPCalibrator, CustomCCP,
                                          GaussianCCP, PolynomialCCP)
from mapie.future.calibrators.ccp.utils import check_required_arguments
from mapie.future.split import SplitCPRegressor

random_state = 1
np.random.seed(random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]

PHI = [
    CustomCCP(lambda X: np.ones((len(X), 1))),
    CustomCCP(None, bias=True),
    CustomCCP([lambda X: X]),
    CustomCCP([lambda X: X, lambda z: z]),
    CustomCCP([lambda X: X, lambda y_pred: y_pred]),
    PolynomialCCP(2, "X", bias=True),
    PolynomialCCP([1, 2], "X", bias=True),
    PolynomialCCP([1, 4, 5], "y_pred", bias=False),
    PolynomialCCP([0, 1, 4, 5], "y_pred", bias=False),
    PolynomialCCP([0, 1, 3], "z", bias=False),
    GaussianCCP(4),
    CustomCCP([lambda X: X, PolynomialCCP(2)]),
    CustomCCP([lambda X: X, GaussianCCP(2)]),
    CustomCCP([
        lambda X: X, PolynomialCCP([1, 2], bias=False)
    ]),
    (lambda X: (X[:, 0] < 3))*CustomCCP([lambda X: X]),
    CustomCCP([lambda X: X])*(lambda X: (X[:, 0] < 3)),
    CustomCCP([lambda X: X])*None,
    CustomCCP([lambda X: X])*(lambda X: (X[:, 0] < 3))*(
        lambda X: (X[:, [0]] > 0)),
]

# n_out without bias
N_OUT = [1, 1, 10, 12, 11, 21, 21, 3, 4, 5, 4, 31, 12, 30, 10, 10, 10, 10]

GAUSS_NEED_FIT_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": 10,
        "sigma": 1,
    },
    {
        "points": 10,
        "sigma": None,
    },
    {
        "points": 10,
        "sigma": None,
        "random_sigma": True,
    },
    {
        "points": 10,
        "sigma": None,
        "random_sigma": False,
    },
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": None,
    },
]

GAUSS_NO_NEED_FIT_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": np.ones(X.shape[1]),
    },
    {
        "points": (np.ones((2, X.shape[1])), [1, 2]),
        "sigma": None,
    },
    {
        "points": (np.ones((2, X.shape[1])), np.ones((2, X.shape[1]))),
        "sigma": None,
    },
]


# ======== CustomCCP =========
@pytest.mark.parametrize("calibrator", PHI)
def test_custom_ccp_calibrator(calibrator: Any) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=calibrator, alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    mapie.predict(X, z=z)


@pytest.mark.parametrize("calibrator, n_out_raw", zip(PHI, N_OUT))
def test_ccp_calibrator_n_attributes(
    calibrator: CCPCalibrator, n_out_raw: int
) -> None:
    """
    Test that the n_in and n_out attributes are corrects
    """
    mapie = SplitCPRegressor(calibrator=clone(calibrator), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    assert mapie.calibrator_.n_in == 10
    assert mapie.calibrator_.n_out == n_out_raw


def test_invalid_multiplication() -> None:
    with pytest.raises(ValueError, match="The function used as multiplier "):
        mapie = SplitCPRegressor(
            calibrator=CustomCCP([lambda X: X])*(
                    lambda X: (X[:, [0, 1]] > 0)),
            alpha=0.1,
        )
        mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("functions", [
    [lambda X, other: X + other, lambda X, other: X - other],
    [lambda X, other: X + other]
])
def test_custom_functions_error(functions: Any) -> None:
    """
    Test that creating a CCPCalibrator object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    for f in functions:     # For coverage
        f(np.ones((10, 1)), np.ones((10, 1)))
    with pytest.raises(
        ValueError,
        match=r"Forbidden required argument in `CustomCCP` calibrator."
    ):
        mapie = SplitCPRegressor(calibrator=CustomCCP(functions), alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("functions", [
    [lambda X, d=1: X + d, lambda X, d=2: X - d],
    [lambda X, c=1, d=1: X + c*d]
])
def test_custom_functions_optional_arg(functions: Any) -> None:
    """
    Test that creating a CCPCalibrator object with functions which have
    optional arguments doesn't raise an error.
    """
    for f in functions:     # For coverage
        f(np.ones((10, 1)))
    mapie = SplitCPRegressor(calibrator=CustomCCP(functions), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


def test_empty_custom_calibrator() -> None:
    """
    Test that creating a CCPCalibrator object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    with pytest.raises(ValueError):
        mapie = SplitCPRegressor(calibrator=CustomCCP([], bias=False),
                                 alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


# ======== PolynomialCCP =========
def test_poly_calibrator_default_init() -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=PolynomialCCP(), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("degree", [2, [0, 1, 3]])
@pytest.mark.parametrize("variable", ["X", "y_pred", "z"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_calibrator_init_other(
    degree: Any, variable: Any, bias: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=PolynomialCCP(
        degree, variable, bias, normalized), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("var", ["other", 1, np.ones((10, 1))])
def test_invalid_variable_value(var: Any) -> None:
    """
    Test that invalid variable value raise error
    """
    with pytest.raises(ValueError):
        mapie = SplitCPRegressor(calibrator=PolynomialCCP(variable=var),
                                 alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


# ======== GaussianCCP =========
def test_gauss_calibrator_default_init() -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=GaussianCCP(), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("points", [3, [X[0, :], X[3, :], X[7, :]],
                                    ([[1], [2], [3]], [1, 2, 3])])
@pytest.mark.parametrize("sigma", [None, 1, list(range(X.shape[1]))])
@pytest.mark.parametrize("random_sigma", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_gauss_init_other(
    points: Any, sigma: Any, random_sigma: Any, bias: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=GaussianCCP(
        points, sigma, random_sigma, bias, normalized), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("points", [np.ones((10)),
                                    np.ones((10, 2, 2))])
def test_invalid_gauss_points(points: Any) -> None:
    """
    Test that invalid ``GaussianCCP`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="Invalid `points` argument."):
        mapie = SplitCPRegressor(calibrator=GaussianCCP(points), alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


def test_invalid_gauss_points_2() -> None:
    """
    Test that invalid ``GaussianCCP`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="There should have as many points"):
        mapie = SplitCPRegressor(calibrator=GaussianCCP(
            points=(np.ones((10, 3)), np.ones((8, 3)))), alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


def test_invalid_gauss_points_3() -> None:
    """
    Test that invalid ``GaussianCCP`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="The standard deviation 2D array"):
        mapie = SplitCPRegressor(calibrator=GaussianCCP(
            points=(np.ones((10, 3)), np.ones((10, 2)))), alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("sigma", ["1",
                                   np.ones((10, 2)),
                                   np.ones((8, 1)),
                                   np.ones(8)])
def test_invalid_gauss_sigma(sigma: Any) -> None:
    """
    Test that invalid ``GaussianCCP`` ``sigma``argument values raise an
    error
    """
    with pytest.raises(ValueError):
        mapie = SplitCPRegressor(calibrator=GaussianCCP(3, sigma),
                                 alpha=0.1)
        mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("ind", range(len(GAUSS_NEED_FIT_SETTINGS)))
def test_gauss_need_calib(ind: int) -> None:
    """
    Test that ``GaussianCCP`` arguments that require later completion
    have ``_need_x_calib`` = ``True``
    """
    mapie = SplitCPRegressor(calibrator=GaussianCCP(
        **GAUSS_NEED_FIT_SETTINGS[ind]), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    check_is_fitted(mapie.calibrator_, mapie.calibrator_.fit_attributes)


@pytest.mark.parametrize("ind", range(len(GAUSS_NO_NEED_FIT_SETTINGS)))
def test_gauss_no_need_calib(ind: int) -> None:
    """
    Test that ``GaussianCCP`` arguments that don't require later
    completion have ``_need_x_calib`` = ``False``
    """
    mapie = SplitCPRegressor(calibrator=GaussianCCP(
        **GAUSS_NEED_FIT_SETTINGS[ind]), alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    check_is_fitted(mapie.calibrator_, mapie.calibrator_.fit_attributes)


@pytest.mark.parametrize("arg1", ["a", None, 1])
@pytest.mark.parametrize("arg2", ["a", None, 1])
def test_check_required_arguments(arg1: Any, arg2: Any) -> None:
    """
    Test that a ValueError is raised if any of the given argument is ``None``.
    """
    if arg1 is None or arg2 is None:
        with pytest.raises(ValueError):
            check_required_arguments(arg1, arg2)
    else:
        check_required_arguments(arg1, arg2)


@pytest.mark.parametrize("calibrator", [
    GaussianCCP(20)*(lambda X: X[:, 0] > 0),
    (lambda X: X > 0)*GaussianCCP(20),
])
def test_gaussian_sampling_with_multiplier(calibrator: CCPCalibrator):
    """
    Test that the points sampled (for the gaussian centers), are sampled
    within the points which have a not null multiplier value
    """
    mapie = SplitCPRegressor(calibrator=calibrator, alpha=0.1)
    mapie.fit(np.linspace(-100, 100, 1000).reshape(-1, 1), np.ones(1000))

    assert all(mapie.calibrator_.points_[i] > 0 for i in range(20))


@pytest.mark.parametrize("calibrator", [
    GaussianCCP(15)*(lambda X: X[:, 0] > 0),
])
def test_gaussian_sampling_error_not_enough_points(calibrator: CCPCalibrator):
    """
    Test that the calibration samples with a not null multiplier value
    to sample the ``points`` points.
    """
    mapie = SplitCPRegressor(calibrator=calibrator, alpha=0.1,
                             cv=ShuffleSplit(1, test_size=0.5))

    with pytest.raises(ValueError, match="There are not enough samples with"):
        mapie.fit(np.linspace(-10, 10, 40).reshape(-1, 1), np.ones(40))


@pytest.mark.parametrize("calibrator", [
    GaussianCCP(30),
])
def test_gaussian_sampling_error_not_enough_points2(calibrator: CCPCalibrator):
    """
    Test that the calibration samples to sample the ``points`` points.
    """
    mapie = SplitCPRegressor(calibrator=calibrator, alpha=0.1,
                             cv=ShuffleSplit(1, test_size=0.5))

    with pytest.raises(ValueError, match="There is not enough valid samples"):
        mapie.fit(np.linspace(-10, 10, 40).reshape(-1, 1), np.ones(40))
