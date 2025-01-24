from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mapie.future.calibrators import StandardCalibrator
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.future.split import SplitCPRegressor
from mapie.regression import MapieRegressor

random_state = 1
np.random.seed(random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]


@pytest.mark.parametrize("sym", [True, False])
def test_calibrator_fit(sym: bool) -> None:
    """Test that calibrator has correct sym parameter"""
    mapie = SplitCPRegressor(calibrator=StandardCalibrator(), alpha=0.1,
                             conformity_score=AbsoluteConformityScore(sym=sym))
    mapie.fit(X, y, calib_kwargs={"z": z})
    assert mapie.calibrator_.sym == sym


@pytest.mark.parametrize("sym", [True, False])
def test_calibrator_fit_predict(sym: bool) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=StandardCalibrator(), alpha=0.1,
                             conformity_score=AbsoluteConformityScore(sym=sym))
    mapie.fit(X, y, calib_kwargs={"z": z})
    mapie.predict(X, z=z)


def test_standard_equivalence() -> None:
    """
    Check that ``SplitCPRegressor`` with ``StandardCalibrator`` gives the
    same results as ``MapieRegressor`` with ``method='base'``.
    """
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    predictor = LinearRegression().fit(X_train, y_train)
    mapie_ccp = SplitCPRegressor(predictor, calibrator=StandardCalibrator(),
                                 cv="prefit", alpha=0.1)
    mapie_ccp.fit(X_calib, y_calib)
    y_pred_ccp, y_pi_ccp = mapie_ccp.predict(X)

    mapie_split = MapieRegressor(predictor, method="base", cv="prefit")
    mapie_split.fit(X_calib, y_calib)
    y_pred_split, y_pi_split = mapie_split.predict(X, alpha=0.1)

    np.testing.assert_allclose(y_pred_ccp, y_pred_split)
    np.testing.assert_allclose(y_pi_ccp, y_pi_split)
