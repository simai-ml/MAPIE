from mapie.conformity_scores import (
    AbsoluteConformityScore,
    GammaConformityScore,
    ResidualNormalisedScore,
)

REGRESSION_CONFORMITY_SCORES_STRING_MAP = {
    "absolute": AbsoluteConformityScore,
    "gamma": GammaConformityScore,
    "residual_normalized": ResidualNormalisedScore,
}
