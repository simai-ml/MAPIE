import warnings
from typing import Union, List, Tuple, cast

from numpy import array
from mapie._typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator


def transform_confidence_level_to_alpha_list(
    confidence_level: Union[float, List[float]]
) -> List[float]:
    if isinstance(confidence_level, float):
        confidence_levels = [confidence_level]
    else:
        confidence_levels = confidence_level
    return [1 - level for level in confidence_levels]


def check_method_not_naive(method: str) -> None:
    if method == "naive":
        raise ValueError(
            '"naive" method not available in MAPIE >= v1'
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


def make_intervals_single_if_single_alpha(
    intervals: NDArray,
    alphas: List[float]
) -> NDArray:
    if len(alphas) == 1:
        return intervals[:, :, 0]
    return intervals


def cast_point_predictions_to_ndarray(
    point_predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> NDArray:
    # This will be useless when we split .predict and .predict_set in back-end
    return cast(NDArray, point_predictions)
