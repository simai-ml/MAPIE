from inspect import signature
from typing import Union

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mapie.calibration import MapieCalibrator
from mapie.metrics import expected_calibration_error, top_label_ece

random_state = 20

CALIBRATORS = [
    "sigmoid", "isotonic", _SigmoidCalibration(), LinearRegression()
]

ESTIMATORS = [
    LogisticRegression(),
    RandomForestClassifier(random_state=random_state),
]

results = {
    "split": {
        "y_score": [
            [np.nan, 0.33333333, np.nan],
            [0.66666667, np.nan, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, np.nan, 0.35635314],
            [np.nan, np.nan, 0.18501723],
        ],
        "top_label_ece": 0.3881,
    },
    "prefit": {
        "y_score": [
            [np.nan, np.nan, 0.85714286],
            [0.83333333, np.nan, np.nan],
            [np.nan, 0.83333333, np.nan],
            [np.nan, np.nan, 0.85714286],
            [np.nan, np.nan, 0.85714286],
            [np.nan, np.nan, 0.85714286],
            [0.83333333, np.nan, np.nan]
        ],
        "top_label_ece": 0.31349206349206343
    }
}

results_binary = {
    "split": {
        "y_score": [
            [0.76226014, np.nan],
            [0.39557708, np.nan],
            [np.nan, 0.66666667],
            [0.75506701, np.nan],
            [np.nan, 0.66666667],
            [0.81175724, np.nan],
            [0.77294068, np.nan],
            [0.62599563, np.nan],
            [np.nan, 0.66666667],
            [np.nan, 0.66666667],
        ],
        "top_label_ece": 0.30562,
        "ece": 0.56657,
    },
    "prefit": {
        "y_score": [
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [np.nan, 0.85714286],
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [0.85714286, np.nan],
            [0.85714286, np.nan],
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [np.nan, 0.85714286]
        ],
        "top_label_ece": 0.1428571428571429,
        "ece": 0.3571428571428571
    },
}


X, y = make_classification(
    n_samples=20,
    n_classes=3,
    n_informative=4,
    random_state=random_state
)

X_, X_test, y_, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_, y_, test_size=0.33, random_state=random_state
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieCalibrator()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_cal = MapieCalibrator()
    assert mapie_cal.method == "top_label"
    assert mapie_cal.calibrator is None
    assert mapie_cal.cv == "split"


def test_default_fit_params() -> None:
    """Test default sample weights and other parameters."""
    mapie_cal = MapieCalibrator()
    assert (
        signature(mapie_cal.fit).parameters["sample_weight"].default
        is None
    )
    assert (
        signature(mapie_cal.fit).parameters["calib_size"].default
        == 0.33
    )
    assert (
        signature(mapie_cal.fit).parameters["random_state"].default
        is None
    )
    assert (
        signature(mapie_cal.fit).parameters["shuffle"].default
        is True
    )
    assert (
        signature(mapie_cal.fit).parameters["stratify"].default
        is None
    )


def test_false_str_estimator() -> None:
    """Test invalid string input for calibrator."""
    with pytest.raises(
        ValueError,
        match=r".*Please provide a string in*",
    ):
        mapie_cal = MapieCalibrator(
            calibrator="not_estimator"
        )
        mapie_cal.fit(X, y)


def test_estimator_none() -> None:
    """Test that no input for calibrator will return a sigmoid"""
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(X, y)
    assert isinstance(
        mapie_cal.calibrators[list(mapie_cal.calibrators.keys())[0]],
        _SigmoidCalibration
    )


def test_check_type_of_target() -> None:
    """Test the type of target."""
    X = [0.5, 0.2, 0.4, 0.8, 3.8]
    y = [0.4, 0.2, 3.6, 3, 0.2]
    mapie_cal = MapieCalibrator()
    with pytest.raises(
        ValueError,
        match=r".*Make sure to have one of the allowed targets:*"
    ):
        mapie_cal.fit(X, y)


def test_other_methods() -> None:
    """Test that invalid string for method returns error"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid method, allowed method are*",
    ):
        mapie_cal = MapieCalibrator(method="no_method")
        mapie_cal.fit(X, y)


def test_prefit_cv_argument() -> None:
    """Test that prefit method works"""
    est = RandomForestClassifier().fit(X, y)
    mapie_cal = MapieCalibrator(estimator=est, cv="prefit")
    mapie_cal.fit(X, y)


def test_split_cv_argument() -> None:
    """Test that split method works"""
    mapie_cal = MapieCalibrator(cv="split")
    mapie_cal.fit(X, y)


@pytest.mark.parametrize("cv", ["noprefit", "nosplit"])
def test_invalid_cv_argument(cv: str) -> None:
    """Test that other cv method does not work"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv argument*",
    ):
        mapie_cal = MapieCalibrator(cv=cv)
        mapie_cal.fit(X, y)


def test_prefit_split_same_results() -> None:
    """Test that prefit and split method return the same result"""
    est = RandomForestClassifier(
        random_state=random_state
    ).fit(X_train, y_train)
    mapie_cal_prefit = MapieCalibrator(estimator=est, cv="prefit")
    mapie_cal_prefit.fit(X_calib, y_calib)

    mapie_cal_split = MapieCalibrator(
        estimator=RandomForestClassifier(random_state=random_state)
    )
    mapie_cal_split.fit(
        X_, y_, random_state=random_state
    )
    y_prefit = mapie_cal_prefit.predict_proba(X_test)
    y_split = mapie_cal_split.predict_proba(X_test)
    np.testing.assert_allclose(y_split, y_prefit)


def test_not_seen_calibrator() -> None:
    """
    Test that there is a warning if no calibration occurs
    due to no calibrator for this class.
    """
    with pytest.warns(
        UserWarning,
        match=r".*WARNING: This predicted label*"
    ):
        mapie_cal = MapieCalibrator()
        mapie_cal.fit(X, y)
        mapie_cal.calibrators.clear()
        mapie_cal.predict_proba(X)


@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_shape_of_output(
    calibrator: Union[str, RegressorMixin],
    estimator: ClassifierMixin
) -> None:
    """Test that the size of the outputs are coherent."""
    mapie_cal = MapieCalibrator(
        estimator=estimator,
        calibrator=calibrator,
    )
    mapie_cal.fit(X, y)
    calib_ = mapie_cal.predict_proba(X)
    assert calib_.shape == (len(y), mapie_cal.n_classes_)


def test_number_of_classes_equal_calibrators() -> None:
    """
    Test that the number of calibrators is the same as the number
    of classes in the calibration step.
    """
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(
        X=X_,
        y=y_,
        random_state=random_state
    )
    y_pred_calib_set = mapie_cal.single_estimator_.predict(X=X_calib)
    assert len(mapie_cal.calibrators) == len(np.unique(y_pred_calib_set))


def test_same_predict() -> None:
    """Test that the same prediction is made regardless of the calibration."""
    mapie_cal = MapieCalibrator(method="top_label")
    mapie_cal.fit(
        X=X_,
        y=y_,
        random_state=random_state
    )
    y_pred_calib_set = mapie_cal.single_estimator_.predict(X=X_test)
    y_pred_calib_set_through_predict = mapie_cal.predict(X=X_test)
    y_pred_calibrated_test_set = np.nanargmax(
        mapie_cal.predict_proba(X=X_test),
        axis=1
    )
    np.testing.assert_allclose(y_pred_calib_set, y_pred_calibrated_test_set)
    np.testing.assert_allclose(
        y_pred_calib_set,
        y_pred_calib_set_through_predict
    )


@pytest.mark.parametrize("cv", MapieCalibrator.valid_cv)
def test_correct_results(cv: str) -> None:
    """
    Test that the y_score and top label score from the test dataset result
    in the correct scores (in a multi-class setting).
    """
    mapie_cal = MapieCalibrator(cv=cv)
    mapie_cal.fit(
        X=X_,
        y=y_,
        random_state=random_state
    )
    pred_ = mapie_cal.predict_proba(X_test)
    top_label_ece_ = top_label_ece(y_test, pred_)
    np.testing.assert_array_almost_equal(
        results[cv]["y_score"], pred_  # type:ignore
    )
    np.testing.assert_allclose(  # type:ignore
        results[cv]["top_label_ece"],
        top_label_ece_,
        rtol=1e-2
    )


@pytest.mark.parametrize("cv", MapieCalibrator.valid_cv)
def test_correct_results_binary(cv: str) -> None:
    """
    Test that the y_score and top label score from the test dataset result
    in the correct scores (in a binary setting).
    """
    X_binary, y_binary = make_classification(
        n_samples=10,
        n_classes=2,
        n_informative=4,
        random_state=random_state
    )
    mapie_cal = MapieCalibrator(cv=cv)
    mapie_cal.fit(
        X=X_binary,
        y=y_binary,
        random_state=random_state
    )
    pred_ = mapie_cal.predict_proba(X_binary)
    top_label_ece_ = top_label_ece(y_binary, pred_)
    ece = expected_calibration_error(y_binary, pred_)
    np.testing.assert_array_almost_equal(
        results_binary[cv]["y_score"], pred_  # type:ignore
    )
    np.testing.assert_allclose(  # type:ignore
        results_binary[cv]["top_label_ece"],
        top_label_ece_,
        rtol=1e-2
    )
    np.testing.assert_allclose(  # type:ignore
        results_binary[cv]["ece"],
        ece,
        rtol=1e-2
    )


def test_different_binary_y_combinations() -> None:
    """
    Test that despite the different maximum in y value, the
    scores are always the same.
    """
    X_comb, y_comb = make_classification(
        n_samples=20,
        n_classes=3,
        n_informative=4,
        random_state=random_state
    )
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(X_comb, y_comb, random_state=random_state)
    y_score = mapie_cal.predict_proba(X_comb)

    y_comb1 = np.where(y_comb == 2, 3, y_comb)
    mapie_cal1 = MapieCalibrator()
    mapie_cal1.fit(X_comb, y_comb1, random_state=random_state)
    y_score1 = mapie_cal1.predict_proba(X_comb)

    y_comb2 = np.where(y_comb == 2, 40, y_comb)
    mapie_cal2 = MapieCalibrator()
    mapie_cal2.fit(X_comb, y_comb2, random_state=random_state)
    y_score2 = mapie_cal2.predict_proba(X_comb)
    np.testing.assert_array_almost_equal(y_score, y_score1)
    np.testing.assert_array_almost_equal(y_score, y_score2)
    assert top_label_ece(
        y_comb, y_score, classes=mapie_cal.classes_
    ) == top_label_ece(
        y_comb1, y_score1, classes=mapie_cal1.classes_
    )
    assert top_label_ece(
        y_comb, y_score, classes=mapie_cal.classes_
    ) == top_label_ece(
        y_comb2, y_score2, classes=mapie_cal2.classes_
    )


@pytest.mark.parametrize(
    "calibrator", [LinearRegression(), "isotonic"]
)
def test_results_with_constant_sample_weights(
    calibrator: Union[str, RegressorMixin]
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.

    Note that the calibration implementations from sklearn `calibration.py`
    file would not pass these tests.
    """
    n_samples = len(X)
    estimator = RandomForestClassifier(random_state=random_state)
    mapie_clf0 = MapieCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf1 = MapieCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf2 = MapieCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf0.fit(X, y, sample_weight=None, random_state=random_state)
    mapie_clf1.fit(
        X, y, sample_weight=np.ones(shape=n_samples),
        random_state=random_state
    )
    mapie_clf2.fit(
        X, y, sample_weight=np.ones(shape=n_samples) * 5,
        random_state=random_state
    )
    y_pred0 = mapie_clf0.predict_proba(X)
    y_pred1 = mapie_clf1.predict_proba(X)
    y_pred2 = mapie_clf2.predict_proba(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred0, y_pred2)


def test_pipeline_compatibility() -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 0])
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
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(X, y)
    mapie = MapieCalibrator(estimator=pipe)
    mapie.fit(X, y)
    mapie.predict(X)
