from typing import Union, List


def transform_confidence_level_to_alpha_list(
    confidence_level: Union[float, List[float]]
) -> List[float]:
    if isinstance(confidence_level, float):
        confidence_levels = [confidence_level]
    else:
        confidence_levels = confidence_level
    return [1 - level for level in confidence_levels]
