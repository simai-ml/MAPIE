"""
===================================================================
[Pre-v1] Plotting CQR with symmetric argument
===================================================================
**Note: we recently released MAPIE v1.0.0, which introduces breaking API changes.**
**This notebook hasn't been updated to the new API yet.**

An example plot of :class:`~mapie.quantile_regression.MapieQuantileRegressor`
illustrating the impact of the symmetry parameter.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor

random_state = 2

##############################################################################
# We generate a synthetic data.

X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

# Define alpha level
alpha = 0.2

# Fit a Gradient Boosting Regressor for quantile regression
gb_reg = GradientBoostingRegressor(
    loss="quantile", alpha=0.5, random_state=random_state
)

# MAPIE Quantile Regressor
mapie_qr = MapieQuantileRegressor(estimator=gb_reg, alpha=alpha)
mapie_qr.fit(X, y, random_state=random_state)
y_pred_sym, y_pis_sym = mapie_qr.predict(X, symmetry=True)
y_pred_asym, y_pis_asym = mapie_qr.predict(X, symmetry=False)
y_qlow = mapie_qr.estimators_[0].predict(X)
y_qup = mapie_qr.estimators_[1].predict(X)

# Calculate coverage scores
coverage_score_sym = regression_coverage_score(
    y, y_pis_sym[:, 0], y_pis_sym[:, 1]
)
coverage_score_asym = regression_coverage_score(
    y, y_pis_asym[:, 0], y_pis_asym[:, 1]
)

# Sort the values for plotting
order = np.argsort(X[:, 0])
X_sorted = X[order]
y_pred_sym_sorted = y_pred_sym[order]
y_pis_sym_sorted = y_pis_sym[order]
y_pred_asym_sorted = y_pred_asym[order]
y_pis_asym_sorted = y_pis_asym[order]
y_qlow = y_qlow[order]
y_qup = y_qup[order]

##############################################################################
# We will plot the predictions and prediction intervals for both symmetric
# and asymmetric intervals. The line represents the predicted values, the
# dashed lines represent the prediction intervals, and the shaded area
# represents the symmetric and asymmetric prediction intervals.

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X, y, alpha=0.3)
plt.plot(X_sorted, y_qlow, color="C1")
plt.plot(X_sorted, y_qup, color="C1")
plt.plot(X_sorted, y_pis_sym_sorted[:, 0], color="C1", ls="--")
plt.plot(X_sorted, y_pis_sym_sorted[:, 1], color="C1", ls="--")
plt.fill_between(
    X_sorted.ravel(),
    y_pis_sym_sorted[:, 0].ravel(),
    y_pis_sym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Symmetric Intervals\n"
    f"Target and effective coverages for "
    f"alpha={alpha:.2f}: ({1-alpha:.3f}, {coverage_score_sym:.3f})"
)

# Plot asymmetric prediction intervals
plt.subplot(1, 2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X, y, alpha=0.3)
plt.plot(X_sorted, y_qlow, color="C2")
plt.plot(X_sorted, y_qup, color="C2")
plt.plot(X_sorted, y_pis_asym_sorted[:, 0], color="C2", ls="--")
plt.plot(X_sorted, y_pis_asym_sorted[:, 1], color="C2", ls="--")
plt.fill_between(
    X_sorted.ravel(),
    y_pis_asym_sorted[:, 0].ravel(),
    y_pis_asym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Asymmetric Intervals\n"
    f"Target and effective coverages for "
    f"alpha={alpha:.2f}: ({1-alpha:.3f}, {coverage_score_asym:.3f})"
)
plt.tight_layout()
plt.show()

##############################################################################
# The symmetric intervals (`symmetry=True`) use a combined set of residuals
# for both bounds, while the asymmetric intervals use distinct residuals for
# each bound, allowing for more flexible and accurate intervals that reflect
# the heteroscedastic nature of the data. The resulting effective coverages
# demonstrate the theoretical guarantee of the target coverage level
# ``1 - α``.
