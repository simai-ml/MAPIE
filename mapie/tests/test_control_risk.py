"""
Testing for control_risk module.
Testing for now risks for multilabel classification
"""
from typing import List, Optional, Union

import numpy as np
import pytest

from mapie._typing import NDArray
from mapie.control_risk.ltt import find_lambda_control_star, ltt_procedure
from mapie.control_risk.p_values import compute_hoeffdding_bentkus_p_value
from mapie.control_risk.risks import (compute_risk_precision,
                                      compute_risk_recall)

lambdas = np.array([0.5, 0.9])

y_toy = np.stack([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
])

y_preds_proba = np.stack([
    [0.2, 0.6, 0.9],
    [0.8, 0.2, 0.6],
    [0.4, 0.8, 0.1],
    [0.6, 0.8, 0.7]
])

y_preds_proba = np.expand_dims(y_preds_proba, axis=2)

test_recall = np.array([
    [1/2, 1.],
    [1., 1.],
    [1/2, 1.],
    [0., 1.]
])

test_precision = np.array([
    [1/2, 1.],
    [1., 1.],
    [0., 1.],
    [0., 1.]
])

r_hat = np.array([0.5, 0.8])

n = 1100

alpha = np.array([0.6])

valid_index = [[0, 1]]

wrong_alpha = 0

wrong_alpha_shape = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

random_state = 42
prng = np.random.RandomState(random_state)


def test_compute_recall_equal() -> None:
    """Test that compute_recall give good result"""
    recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall, test_recall)


def test_compute_precision() -> None:
    """Test that compute_precision give good result"""
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision, test_precision)


def test_recall_with_zero_sum_is_equal_nan() -> None:
    """Test compute_recall with nan values"""
    y_toy = np.zeros((4, 3))
    y_preds_proba = prng.rand(4, 3, 1)
    recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(recall, np.empty_like(recall))


def test_precision_with_zero_sum_is_equal_ones() -> None:
    """Test compute_precision with nan values"""
    y_toy = prng.rand(4, 3)
    y_preds_proba = np.zeros((4, 3, 1))
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(precision, np.ones_like(precision))


def test_compute_recall_shape() -> None:
    """Test shape when using _compute_recall"""
    recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall.shape, test_recall.shape)


def test_compute_precision_shape() -> None:
    """Test shape when using _compute_precision"""
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision.shape, test_precision.shape)


def test_compute_recall_with_wrong_shape() -> None:
    """Test error when wrong shape in _compute_recall"""
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        compute_risk_recall(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        compute_risk_recall(lambdas, y_preds_proba, np.expand_dims(y_toy, 2))
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        compute_risk_recall(lambdas, y_preds_proba, y_toy[:-1])


def test_compute_precision_with_wrong_shape() -> None:
    """Test shape when using _compute_precision"""
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        compute_risk_precision(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        compute_risk_precision(
            lambdas, y_preds_proba, np.expand_dims(y_toy, 2)
        )
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        compute_risk_precision(lambdas, y_preds_proba, y_toy[:-1])


@pytest.mark.parametrize("alpha", [0.5, [0.5], [0.5, 0.9]])
def test_p_values_different_alpha(alpha: Union[float, NDArray]) -> None:
    """Test type for different alpha for p_values"""
    result = compute_hoeffdding_bentkus_p_value(r_hat, n, alpha)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("delta", [0.1, 0.2])
def test_ltt_different_delta(delta: float) -> None:
    """Test _ltt_procedure for different delta"""
    assert ltt_procedure(r_hat, alpha, delta, n)


def test_find_lambda_control_star() -> None:
    """Test _find_lambda_control_star"""
    assert find_lambda_control_star(r_hat, valid_index, lambdas)


@pytest.mark.parametrize("delta", [0.1, 0.8])
@pytest.mark.parametrize("alpha", [[0.5], [0.6, 0.8]])
def test_ltt_type_output_alpha_delta(
    alpha: NDArray,
    delta: float
) -> None:
    """Test type output _ltt_procedure"""
    valid_index, p_values = ltt_procedure(r_hat, alpha, delta, n)
    assert isinstance(valid_index, list)
    assert isinstance(p_values, np.ndarray)


@pytest.mark.parametrize("valid_index", [[[0, 1]]])
def test_find_lambda_control_star_output(valid_index: List[List[int]]) -> None:
    """Test _find_lambda_control_star with a list of list"""
    assert find_lambda_control_star(r_hat, valid_index, lambdas)


def test_warning_valid_index_empty() -> None:
    """Test warning sent when empty list"""
    valid_index = [[]]  # type: List[List[int]]
    with pytest.warns(
        UserWarning, match=r".*At least one sequence is empty*"
    ):
        find_lambda_control_star(r_hat, valid_index, lambdas)


def test_invalid_alpha_hb() -> None:
    """Test error message when invalid alpha"""
    with pytest.raises(ValueError, match=r".*Invalid confidence_level"):
        compute_hoeffdding_bentkus_p_value(r_hat, n, wrong_alpha)


def test_invalid_shape_alpha_hb() -> None:
    """Test error message when invalid alpha shape"""
    with pytest.raises(ValueError, match=r".*Invalid confidence_level"):
        compute_hoeffdding_bentkus_p_value(r_hat, n, wrong_alpha_shape)


@pytest.mark.parametrize("delta", [None])
def test_delta_none_ltt(delta: Optional[float]) -> None:
    """Test error message when invalid delta"""
    with pytest.raises(ValueError, match=r".*Invalid delta"):
        ltt_procedure(r_hat, alpha, delta, n)
