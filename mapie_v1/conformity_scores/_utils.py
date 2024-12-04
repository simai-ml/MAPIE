from typing import Union
from mapie.conformity_scores import BaseRegressionScore
from . import REGRESSION_CONFORMITY_SCORES_STRING_MAP


def check_and_select_regression_conformity_score(
    conformity_score: Union[str, BaseRegressionScore]
) -> BaseRegressionScore:
    if isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    elif conformity_score in REGRESSION_CONFORMITY_SCORES_STRING_MAP:
        return REGRESSION_CONFORMITY_SCORES_STRING_MAP[conformity_score]()
    else:
        raise ValueError("Invalid conformity_score type")
