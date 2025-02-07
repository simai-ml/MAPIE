from __future__ import annotations

from inspect import signature
from typing import Any, Callable, cast

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     PredefinedSplit, RepeatedKFold,
                                     ShuffleSplit, TimeSeriesSplit,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

from mapie._typing import NDArray
from mapie.future.calibrators.ccp import (CCPCalibrator, CustomCCP,
                                          GaussianCCP, PolynomialCCP)
from mapie.conformity_scores import LACConformityScore, APSConformityScore
from mapie.conformity_scores import BaseClassificationScore
from mapie.metrics import classification_coverage_score
from mapie.future.split import SplitCPClassifier

random_state = 1
np.random.seed(random_state)

N_CLASSES = 4
X, y = make_classification(
    n_samples=200, n_features=10,
    n_informative=N_CLASSES, n_classes=N_CLASSES,
    random_state=random_state
)
z = X[:, -2:]

CV = ["prefit", "split"]

PHI = [
    CustomCCP([lambda X: np.ones((len(X), 1))]),
    PolynomialCCP([0, 1]),
    GaussianCCP(5),
]
WIDTHS = {
    "split": 1.835,
    "prefit": 1.835,
}

COVERAGES = {
    "split": 0.885,
    "prefit": 0.885,
}


# ======== MapieCCPRegressor =========
def test_initialized() -> None:
    """Test that initialization does not crash."""
    SplitCPClassifier(alpha=0.1)


def test_fit_predictor() -> None:
    """Test that fit_predictor raises no errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit_predictor(X, y)


@pytest.mark.parametrize("z", [None, z])
def test_fit_calibrator(z: Any) -> None:
    """Test that fit_calibrator raises no errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit_predictor(X, y)
    mapie.fit_calibrator(X, y, z=z)


@pytest.mark.parametrize("z", [None, z])
def test_fit(z: Any) -> None:
    """Test that fit raises no errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})


@pytest.mark.parametrize("z", [None, z])
def test_fit_predictor_fit_calibrator_predict(z: Any) -> None:
    """Test that fit-calibrate-predict raises no errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit_predictor(X, y)
    mapie.fit_calibrator(X, y, z=z)
    mapie.predict(X, z=z)


@pytest.mark.parametrize("z", [None, z])
def test_fit_predict(z: Any) -> None:
    """Test that fit-predict raises no errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    mapie.predict(X, z=z)


@pytest.mark.parametrize("z", [None, z])
def test_fit_predict_reg(z: Any) -> None:
    """Test that fit-predict raises no errors."""
    mapie = SplitCPClassifier(calibrator=GaussianCCP(reg_param=0.1),
                              alpha=0.1)
    mapie.fit(X, y, calib_kwargs={"z": z})
    mapie.predict(X, z=z)


def test_not_fitted_predictor_fit_calibrator() -> None:
    """Test that calibrate before fit raises errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie.fit_calibrator(X, y)


def test_calib_not_complete_phi() -> None:
    """Test that a not complete calibrator definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie = SplitCPClassifier(
            alpha=0.1,
            calibrator=CustomCCP([lambda X: (X[:, 0] > 0).astype(int)],
                                 bias=False)
        )
        mapie.fit(X, y)


def test_predict_not_complete_phi() -> None:
    """Test that a not complete calibrator definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie = SplitCPClassifier(
            alpha=0.1,
            calibrator=CustomCCP([lambda X: (X[:, 0] > 0).astype(int)],
                                 bias=False)
        )
        mapie.fit(X[X[:, 0] < 0], y[X[:, 0] < 0])
        mapie.predict(X)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie.predict(X)


def test_no_calibrate_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie = SplitCPClassifier(alpha=0.1)
    mapie.fit_predictor(X, y)
    with pytest.raises(NotFittedError):
        mapie.predict(X)


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie = SplitCPClassifier(alpha=0.1)
    assert (
        signature(mapie.fit_predictor).parameters["sample_weight"].default
        is None
    )


@pytest.mark.parametrize("predictor", [0, "a", KFold(), ["a", "b"]])
def test_invalid_predictor(
    predictor: Any
) -> None:
    """Test that invalid predictors raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie = SplitCPClassifier(predictor=predictor, alpha=0.1)
        mapie.fit_predictor(X, y)


@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_invalid_prefit_predictor_calibrate(
    predictor: ClassifierMixin,
) -> None:
    """Test that non-fitted predictor with prefit cv raise errors when
    calibrate is called"""
    with pytest.raises(NotFittedError):
        mapie = SplitCPClassifier(predictor=predictor, cv="prefit",
                                  alpha=0.1)
        mapie.fit_calibrator(X, y)


@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_invalid_prefit_predictor_fit(
    predictor: ClassifierMixin,
) -> None:
    """Test that non-fitted predictor with prefit cv raise errors when fit
    is called."""
    with pytest.raises(NotFittedError):
        mapie = SplitCPClassifier(predictor=predictor, cv="prefit",
                                  alpha=0.1)
        mapie.fit_predictor(X, y)


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = SplitCPClassifier(random_state=random_state, alpha=0.1)
    mapie.fit(X, y)
    assert isinstance(mapie.predictor_, ClassifierMixin)
    assert isinstance(mapie.calibrator_, GaussianCCP)
    assert isinstance(mapie.cv, ShuffleSplit)
    assert mapie.alpha == 0.1
    assert isinstance(mapie.conformity_score_, BaseClassificationScore)
    assert isinstance(mapie.random_state, int)


@pytest.mark.parametrize(
    "alpha", ["a", 0, 2, 1.5, -0.3]
)
def test_invalid_alpha(alpha: Any) -> None:
    with pytest.raises(ValueError):
        mapie = SplitCPClassifier(alpha=alpha)
        mapie.fit(X, y)


@pytest.mark.parametrize(
    "calibrator", [1, "some_string"]
)
def test_invalid_phi(calibrator: Any) -> None:
    with pytest.raises(ValueError):
        mapie = SplitCPClassifier(calibrator=calibrator)
        mapie.fit(X, y)


def test_valid_predictor() -> None:
    """Test that valid predictors are not corrupted"""
    mapie = SplitCPClassifier(
        predictor=DummyClassifier(),
        random_state=random_state,
        alpha=0.1,
    )
    mapie.fit_predictor(X, y)
    assert isinstance(mapie.predictor, DummyClassifier)


@pytest.mark.parametrize(
    "cv", [None, ShuffleSplit(n_splits=1),
           PredefinedSplit(
                test_fold=[1]*(len(X)//2) + [-1]*(len(X)-len(X)//2)
           ), "prefit", "split"]
)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_valid_cv(cv: Any, predictor: ClassifierMixin) -> None:
    """Test that valid cv raise no errors."""
    predictor.fit(X, y)
    mapie = SplitCPClassifier(predictor, CustomCCP(bias=True), cv=cv,
                              alpha=0.1, random_state=random_state)
    mapie.fit(X, y)
    mapie.predict(X)


@pytest.mark.parametrize(
    "cv", ["dummy", 0, 1, 1.5] + [  # Cross val splitters
            3, -1, KFold(n_splits=5), LeaveOneOut(),
            RepeatedKFold(n_splits=5, n_repeats=2), ShuffleSplit(n_splits=5),
            TimeSeriesSplit(), LeavePOut(p=2),
            PredefinedSplit(test_fold=[0]*(len(X)//4) + [1]*(len(X)//4)
                            + [-1]*(len(X)-len(X)//2)),
           ]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid agg_functions raise errors."""
    with pytest.raises(ValueError, match="Invalid cv argument."):
        mapie = SplitCPClassifier(cv=cv, alpha=0.1,
                                  random_state=random_state)
        mapie.fit_predictor(X, y)


@pytest.mark.parametrize("alpha", [0.2])
@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_fit_calibrate_combined_equivalence(
    alpha: Any, cv: Any, calibrator: CCPCalibrator, predictor: ClassifierMixin
) -> None:
    """Test predict output shape."""
    predictor_1 = clone(predictor)
    predictor_2 = clone(predictor)
    if cv == "prefit":
        predictor_1.fit(X, y)
        predictor_2.fit(X, y)

    np.random.seed(random_state)
    mapie_1 = SplitCPClassifier(
        predictor=predictor_1, calibrator=calibrator,
        cv=cv, alpha=alpha, random_state=random_state
    )
    np.random.seed(random_state)
    mapie_2 = SplitCPClassifier(
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


@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_predict_output_shape_alpha(
    cv: Any, calibrator: CCPCalibrator, predictor: ClassifierMixin
) -> None:
    """Test predict output shape."""
    if cv == "prefit":
        predictor.fit(X, y)

    mapie = SplitCPClassifier(
        predictor=predictor, calibrator=calibrator,
        cv=cv, alpha=0.1, random_state=random_state
    )
    mapie.fit(X, y, calib_kwargs={"z": z})
    y_pred, y_pis = mapie.predict(X, z=z)
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], N_CLASSES, 1)


@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_predict_output_shape_no_alpha(
    cv: Any, calibrator: CCPCalibrator, predictor: ClassifierMixin
) -> None:
    """Test predict output shape."""
    if cv == "prefit":
        predictor.fit(X, y)

    mapie = SplitCPClassifier(
        predictor=predictor, calibrator=calibrator, cv=cv,
        alpha=None, random_state=random_state
    )
    mapie.fit(X, y, calib_kwargs={"z": z})
    y_pred = mapie.predict(X, z=z)
    assert np.array(y_pred).shape == (X.shape[0],)


@pytest.mark.parametrize("template", PHI)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_same_results_prefit_split(
    template: CCPCalibrator,
    predictor: ClassifierMixin,
) -> None:
    """
    Test checking that if split and prefit method have exactly
    the same data split, then we have exactly the same results.
    """
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

    mapie_1 = SplitCPClassifier(
        clone(predictor), clone(calibrator), pred_cv, alpha=0.1,
        random_state=random_state,
    )

    fitted_predictor = clone(predictor).fit(X_train, y_train)
    mapie_2 = SplitCPClassifier(
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


@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
def test_results_for_ordered_alpha(
    cv: Any, calibrator: CCPCalibrator, predictor: ClassifierMixin
) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    if cv == "prefit":
        predictor.fit(X, y)

    calibrator._transform_params(X)

    mapie_reg_1 = SplitCPClassifier(predictor, clone(calibrator), cv=cv,
                                    alpha=0.05, random_state=random_state)
    mapie_reg_2 = SplitCPClassifier(predictor, clone(calibrator), cv=cv,
                                    alpha=0.1, random_state=random_state)

    mapie_reg_1.fit(X, y, calib_kwargs={"z": z})
    _, y_pis_1 = mapie_reg_1.predict(X, z=z)
    mapie_reg_2.fit(X, y, calib_kwargs={"z": z})
    _, y_pis_2 = mapie_reg_1.predict(X, z=z)

    assert (y_pis_1[:, 0, 0] <= y_pis_2[:, 0, 0]).all()
    assert (y_pis_1[:, 1, 0] >= y_pis_2[:, 1, 0]).all()


def test_results_split() -> None:
    """Test prefit results on a standard train/validation/test split."""
    cv = ShuffleSplit(1, test_size=0.5, random_state=random_state)
    predictor = LogisticRegression()
    mapie = SplitCPClassifier(
        predictor=predictor, calibrator=clone(PHI[0]), cv=cv, alpha=0.2,
        random_state=random_state
    )
    mapie.fit(X, y)
    _, y_ps = mapie.predict(X)
    width_mean = y_ps.sum(axis=1).mean()
    coverage = classification_coverage_score(y, y_ps[:, :, 0])
    np.testing.assert_allclose(width_mean, WIDTHS["split"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["split"], rtol=1e-2)


def test_results_prefit() -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    predictor = LogisticRegression().fit(X_train, y_train)
    mapie = SplitCPClassifier(
        predictor=predictor, calibrator=clone(PHI[0]), cv="prefit", alpha=0.2,
        random_state=random_state
    )
    mapie.fit(X_calib, y_calib)
    _, y_ps = mapie.predict(X)
    width_mean = y_ps.sum(axis=1).mean()
    coverage = classification_coverage_score(y, y_ps[:, :, 0])
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


@pytest.mark.parametrize("calibrator", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("predictor", [
    LogisticRegression(),
    make_pipeline(LogisticRegression()),
])
@pytest.mark.parametrize(
    "conformity_score", [LACConformityScore(), APSConformityScore()]
)
def test_conformity_score(
    cv: Any,
    calibrator: CCPCalibrator,
    predictor: ClassifierMixin,
    conformity_score: BaseClassificationScore,
) -> None:
    """Test that any conformity score function with MAPIE raises no error."""

    if cv == "prefit":
        predictor.fit(X, y)

    mapie = SplitCPClassifier(
        predictor=predictor,
        calibrator=calibrator,
        cv=cv,
        alpha=0.1,
        conformity_score=conformity_score,
        random_state=random_state,
    )
    mapie.fit(X, y, calib_kwargs={"z": z})
    mapie.predict(X, z=z)


def test_fit_parameters_passing() -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting predictors have used 3 iterations
    only during boosting, instead of default value for n_predictors (=100).
    """
    gb = GradientBoostingClassifier(random_state=random_state)

    mapie = SplitCPClassifier(predictor=gb, alpha=0.1,
                              random_state=random_state)

    def early_stopping_monitor(i, est, locals):
        """Returns True on the 3rd iteration."""
        if i == 2:
            return True
        else:
            return False

    mapie.fit(X, y, fit_kwargs={"monitor": early_stopping_monitor})

    assert cast(ClassifierMixin, mapie.predictor).estimators_.shape[0] == 3


@pytest.mark.parametrize("custom_method", [
    lambda local_arg: local_arg,
    lambda self_arg: self_arg,
    lambda kwarg_arg: kwarg_arg,
    lambda local_arg, *args, **kwargs: local_arg,
    lambda self_arg, *args, **kwargs: self_arg,
    lambda kwarg_arg, *args, **kwargs: kwarg_arg,
])
def test_get_method_arguments(custom_method: Callable) -> None:
    mapie = SplitCPClassifier(alpha=0.1)
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
    mapie = SplitCPClassifier()
    assert mapie._check_conformity_scores(conformity_scores).shape == (200,)


def test_check_conformity_scores_error() -> None:
    mapie = SplitCPClassifier()
    with pytest.raises(ValueError, match="Invalid conformity scores."):
        mapie._check_conformity_scores(np.random.rand(200, 5))


def test_invalid_classifier():
    """
    Fitted classifier must contain the ``classes_`` attribute
    """
    class Custom(ClassifierMixin):
        def __init__(self) -> None:
            self.fitted_ = True

        def fit(self):
            pass

        def predict(self):
            pass

        def predict_proba(self):
            pass

    invalid_cls = Custom()
    # for coverage:
    invalid_cls.fit()
    invalid_cls.predict()
    invalid_cls.predict_proba()

    mapie = SplitCPClassifier(invalid_cls, cv="prefit", alpha=0.1)
    with pytest.raises(AttributeError,
                       match="Fitted classifier must contain 'classes_' attr"):
        mapie.fit(X, y)
