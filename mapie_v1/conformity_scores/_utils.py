from typing import no_type_check
from . import CONFORMITY_SCORES_STRING_MAP


@no_type_check  # Cumbersome to type check
def check_and_select_conformity_score(conformity_score, conformity_score_type):
    if isinstance(conformity_score, conformity_score_type):
        return conformity_score
    elif conformity_score in CONFORMITY_SCORES_STRING_MAP[conformity_score_type]:
        return CONFORMITY_SCORES_STRING_MAP[conformity_score_type][conformity_score]()
    else:
        raise ValueError("Invalid conformity_score parameter")
