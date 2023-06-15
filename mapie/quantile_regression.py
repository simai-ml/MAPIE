from sklearn.utils import deprecated

from mapie.regression import MapieQuantileRegressor as NewClass


@deprecated(
    "WARNING: Deprecated path to import MapieQuantileRegressor. "
    "Please prefer the new path: "
    "[form mapie.regression import MapieQuantileRegressor]."
)
class MapieQuantileRegressor(NewClass):
    pass
