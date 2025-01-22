import warnings
from typing import Union, List, Tuple, cast

from numpy import array
from mapie._typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from decimal import Decimal


def transform_confidence_level_to_alpha_list(
    confidence_level: Union[float, List[float]]
) -> List[float]:
    if isinstance(confidence_level, list):
        confidence_levels = confidence_level
    else:
        confidence_levels = [confidence_level]

    # Using decimals to avoid weird-looking float approximations
    # when computing alpha = 1 - confidence_level
    # Such approximations arise even with simple confidence levels like 0.9
    confidence_levels_d = [Decimal(str(conf_level)) for conf_level in confidence_levels]
    alphas_d = [Decimal("1") - conf_level_d for conf_level_d in confidence_levels_d]

    return [float(alpha_d) for alpha_d in alphas_d]


def check_if_param_in_allowed_values(
    param: str, param_name: str, allowed_values: list
) -> None:
    if param not in allowed_values:
        raise ValueError(
            f"'{param}' option not valid for parameter '{param_name}'"
            f"Available options are: {allowed_values}"
        )


def check_cv_not_string(cv: Union[int, str, BaseCrossValidator]):
    if isinstance(cv, str):
        raise ValueError(
            "'cv' string options not available in MAPIE >= v1"
        )


def hash_X_y(X: ArrayLike, y: ArrayLike) -> int:
    # Known issues:
    # - the hash calculated with `hash` changes between Python processes
    # - two arrays with  the same content but different shapes will all have
    #   the same hash because .tobytes() ignores shape
    return hash(array(X).tobytes() + array(y).tobytes())


def check_if_X_y_different_from_fit(
    X: ArrayLike,
    y: ArrayLike,
    previous_X_y_hash: int
) -> None:
    if hash_X_y(X, y) != previous_X_y_hash:
        warnings.warn(
            "You have to use the same X and y in .fit and .conformalize"
        )


def cast_point_predictions_to_ndarray(
    point_predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> NDArray:
    return cast(NDArray, point_predictions)


def cast_predictions_to_ndarray_tuple(
    predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> Tuple[NDArray, NDArray]:
    return cast(Tuple[NDArray, NDArray], predictions)
