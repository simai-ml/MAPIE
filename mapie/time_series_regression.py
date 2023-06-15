from sklearn.utils import deprecated

from mapie.regression import MapieTimeSeriesRegressor as NewClass


@deprecated(
    "WARNING: Deprecated path to import MapieTimeSeriesRegressor. "
    "Please prefer the new path: "
    "[form mapie.regression import MapieTimeSeriesRegressor]."
)
class MapieTimeSeriesRegressor(NewClass):
    pass
