from typing import Union, List
from mapie.conformity_scores import BaseRegressionScore
from . import REGRESSION_CONFORMITY_SCORES_STRING_MAP


def check_and_select_split_conformity_score(
    conformity_score: Union[str, BaseRegressionScore]
) -> BaseRegressionScore:
    if isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    elif conformity_score in REGRESSION_CONFORMITY_SCORES_STRING_MAP:
        return REGRESSION_CONFORMITY_SCORES_STRING_MAP[conformity_score]()
    else:
        raise ValueError("Invalid conformity_score type")


def process_confidence_level(
    self, confidence_level: Union[float, List[float]]
) -> List[float]:
    """
    Ensure confidence_level is always a list of floats.
    """
    if isinstance(confidence_level, float):
        return [confidence_level]
    return confidence_level


def compute_alpha(
    self, confidence_levels: List[float]
) -> List[float]:
    """
    Compute alpha values from confidence levels.
    """
    return [1 - level for level in confidence_levels]
