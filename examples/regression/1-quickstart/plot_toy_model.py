"""
=====================================================================================
Plotting MAPIE prediction intervals with a toy dataset
=====================================================================================
An example plot of :class:`~mapie_v1.regression.SplitConformalRegressor` used
in the Quickstart.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from mapie.metrics import regression_coverage_score
from mapie_v1.regression import SplitConformalRegressor

RANDOM_STATE = 42
X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=RANDOM_STATE)
X_train_conformalize, X_test, y_train_conformalize, y_test = train_test_split(
    X, y, test_size=0.5, random_state=RANDOM_STATE
)
X_train, X_conformalize, y_train, y_conformalize = train_test_split(
    X_train_conformalize, y_train_conformalize, test_size=0.5, random_state=RANDOM_STATE
)

confidence_level = [0.95, 0.68]
mapie_regressor = SplitConformalRegressor(
    confidence_level=confidence_level, prefit=False
)
mapie_regressor.fit(X_train, y_train)
mapie_regressor.conformalize(X_conformalize, y_conformalize)
y_pred, y_pred_interval = mapie_regressor.predict_interval(X_test)

coverage_scores = [
    regression_coverage_score(
        y_test, y_pred_interval[:, 0, i], y_pred_interval[:, 1, i]
    ) for i, _ in enumerate(confidence_level)
]

plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_test, y_test, alpha=0.3)
plt.plot(X_test, y_pred, color="C1")
order = np.argsort(X_test[:, 0])
plt.plot(X_test[order], y_pred_interval[order][:, 0, 1], color="C1", ls="--")
plt.plot(X_test[order], y_pred_interval[order][:, 1, 1], color="C1", ls="--")
plt.fill_between(
    X_test[order].ravel(),
    y_pred_interval[order][:, 0, 0].ravel(),
    y_pred_interval[order][:, 1, 0].ravel(),
    alpha=0.2,
)
plt.title(
    f"Effective coverage for "
    f"confidence_level={confidence_level[0]:.2f}: {coverage_scores[0]:.3f}\n"
    f"Effective coverage for "
    f"confidence_level={confidence_level[1]:.2f}: {coverage_scores[1]:.3f}"
)
plt.show()
