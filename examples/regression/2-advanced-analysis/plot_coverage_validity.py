"""
===============================================================================
[Pre-v1] Coverage Validity with MAPIE for Regression Task
===============================================================================
**Note: we recently released MAPIE v1.0.0, which introduces breaking API changes.**
**This notebook hasn't been updated to the new API yet.**


This example verifies that conformal claims are valid in the MAPIE package
when using the CP prefit/split methods.

This notebook is inspired of the notebook used for episode "Uncertainty
Quantification: Avoid these Missteps in Validating Your Conformal Claims!"
(link to the [orginal notebook](https://github.com/mtorabirad/MLBoost)).

For more details on theoretical guarantees:

[1] Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer.
"Algorithmic Learning in a Random World." Springer Nature, 2022.

[2] Angelopoulos, Anastasios N., and Stephen Bates.
"Conformal prediction: A gentle introduction." Foundations and Trends®
in Machine Learning 16.4 (2023): 494-591.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import ShuffleSplit, train_test_split

from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.metrics import regression_coverage_score_v2

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)


##############################################################################
# Section 1: Comparison with the split conformalizer method (light version)
# -------------------------------------------------------------------------
#
# We propose here to implement a lighter version of split CP by calculating
# the quantile with a small correction according to [1].
# We prepare the fit/calibration/test routine in order to calculate the average
# coverage over several simulations.

# Conformalizer Class
class StandardConformalizer():
    def __init__(
        self,
        pre_trained_model,
        non_conformity_func,
        delta
    ):
        # Initialize the conformalizer with required parameters
        self.estimator = pre_trained_model
        self.non_conformity_func = non_conformity_func
        self.delta = delta

    def _calculate_quantile(self, scores_calib):
        # Calculate the quantile value based on delta and non-conformity scores
        self.delta_cor = np.ceil(self.delta*(self.n_calib+1))/self.n_calib
        return np.quantile(scores_calib, self.delta_cor, method='lower')

    def _calibrate(self, X_calib, y_calib):
        # Calibrate the conformalizer to calculate q_hat
        y_calib_pred = self.estimator.predict(X_calib)
        scores_calib = self.non_conformity_func(y_calib_pred, y_calib)
        self.q_hat = self._calculate_quantile(scores_calib)

    def fit(self, X, y):
        # Fit the conformalizer to the data and calculate q_hat
        self.n_calib = X.shape[0]
        self._calibrate(X, y)
        return self

    def predict(self, X, alpha=None):
        # Returns the predicted interval
        y_pred = self.estimator.predict(X)
        y_lower, y_upper = y_pred - self.q_hat, y_pred + self.q_hat
        y_pis = np.expand_dims(np.stack([y_lower, y_upper], axis=1), axis=-1)
        return y_lower, y_pis


def non_conformity_func(y, y_hat):
    return np.abs(y - y_hat)


def get_coverage_prefit(
    conformalizer, data, target, delta, n_calib, random_state=None
):
    """
    Calculate the fraction of test samples within the predicted intervals.

    This function splits the data into a training set and a test set. If the
    cross-validation strategy of the mapie regressor is a ShuffleSplit, it fits
    the regressor to the entire training set. Otherwise, it further splits the
    training set into a calibration set and a training set, and fits the
    regressor to the calibration set. It then predicts intervals for the test
    set and calculates the fraction of test samples within these intervals.

    Parameters:
    -----------
    conformalizer: object
        A mapie regressor object.

    data: array-like of shape (n_samples, n_features)
        The data to be split into a training set and a test set.

    target: array-like of shape (n_samples,)
        The target values for the data.

    delta: float
        The level of confidence for the predicted intervals.

    Returns:
    --------
    fraction_within_bounds: float
        The fraction of test samples within the predicted intervals.
    """
    # Split data step
    X_cal, X_test, y_cal, y_test = train_test_split(
        data, target, train_size=n_calib, random_state=random_state
    )
    # Calibration step
    conformalizer.fit(X_cal, y_cal)
    # Prediction step
    _, y_pis = conformalizer.predict(X_test, alpha=1-delta)
    # Coverage step
    coverage = regression_coverage_score_v2(y_test, y_pis)

    return coverage


def cumulative_average(arr):
    """
    Calculate the cumulative average of a list of numbers.

    This function computes the cumulative average of a list of numbers by
    calculating the cumulative sum of the numbers and dividing it by the
    index of the current number.

    Parameters:
    -----------
    arr: List[float]
        The input list of numbers.

    Returns:
    --------
    running_avg: List[float]
        The cumulative average of the input list.
    """
    cumsum = np.cumsum(arr)
    indices = np.arange(1, len(arr) + 1)
    cumulative_avg = cumsum / indices
    return cumulative_avg


##############################################################################
# Experiment 1: Coverage Validity for a given delta, n_calib
# ----------------------------------------------------------
#
# To begin, we propose to use ``delta=0.8`` and ``n_delta=6`` and compare
# the coverage validity claim of the MAPIE class and the referenced class.

# Parameters of the modelisation
delta = 0.8
n_calib = 6

n_train = 1000
n_test = 1000
num_splits = 1000

# Load toy Data
n_all = n_train + n_calib + n_test
data, target = make_regression(n_all, random_state=1)

# Split dataset into training, calibration and validation sets
X_train, X_cal_test, y_train, y_cal_test = train_test_split(
    data, target, train_size=n_train, random_state=1
)

# Create a regression model and fit it to the training data
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Compute theorical bounds and exact coverage to attempt
lower_bound = delta
upper_bound = (delta + 1/(n_calib+1))
upper_bound_2 = (delta + 1/(n_calib/2+1))
exact_cov = (np.ceil((n_calib+1)*delta))/(n_calib+1)

# Run the experiment
empirical_coverages_ref = []
empirical_coverages_mapie = []

for i in range(1, num_splits):
    # Compute empirical coverage for each trial with StandardConformalizer
    conformalizer = StandardConformalizer(model, non_conformity_func, delta)
    coverage = get_coverage_prefit(
        conformalizer, X_cal_test, y_cal_test, delta, n_calib, random_state=i
    )
    empirical_coverages_ref.append(coverage)

    # Compute empirical coverage for each trial with MapieRegressor
    conformalizer = MapieRegressor(estimator=model, cv="prefit")
    coverage = get_coverage_prefit(
        conformalizer, X_cal_test, y_cal_test, delta, n_calib, random_state=i
    )
    empirical_coverages_mapie.append(coverage)

cumulative_averages_ref = cumulative_average(empirical_coverages_ref)
cumulative_averages_mapie = cumulative_average(empirical_coverages_mapie)

# Plot the results
fig, ax = plt.subplots()
plt.plot(cumulative_averages_ref, alpha=0.5, label='SplitCP', color='r')
plt.plot(cumulative_averages_mapie, alpha=0.5, label='MAPIE', color='g')

plt.hlines(exact_cov, 0, num_splits, color='r', ls='--', label='Exact Cov.')
plt.hlines(lower_bound, 0, num_splits, color='k', label='Lower Bound')
plt.hlines(upper_bound, 0, num_splits, color='b', label='Upper Bound')

plt.xlabel(r'Split Number')
plt.ylabel(r'$\overline{\mathbb{C}}$')
plt.title(r'$|D_{cal}| = $' + str(n_calib) + r' and $\delta = $' + str(delta))

plt.legend(loc="upper right", ncol=2)
plt.ylim(0.7, 1)
plt.tight_layout()
plt.show()


##############################################################################
# It can be seen that the two curves overlap, proving that both methods
# produce the same results. Their effective coverage stabilizes between
# the theoretical limits, always above the target coverage and converges
# towards the exact coverage (i.e. expected according to the theory).


##############################################################################
# Experiment 2: Again but without fixing random_state
# ---------------------------------------------------
#
# We just propose to reproduce the previous experiment without fixing the
# random_state. The methods therefore follow different trajectories but
# always achieve the expected coverage.

# Run the experiment
empirical_coverages_ref = []
empirical_coverages_mapie = []

for i in range(1, num_splits):
    # Compute empirical coverage for each trial with StandardConformalizer
    conformalizer = StandardConformalizer(model, non_conformity_func, delta)
    coverage = get_coverage_prefit(
        conformalizer, X_cal_test, y_cal_test, delta, n_calib
    )
    empirical_coverages_ref.append(coverage)

    # Compute empirical coverage for each trial with MapieRegressor
    conformalizer = MapieRegressor(estimator=model, cv="prefit")
    coverage = get_coverage_prefit(
        conformalizer, X_cal_test, y_cal_test, delta, n_calib
    )
    empirical_coverages_mapie.append(coverage)

cumulative_averages_ref = cumulative_average(empirical_coverages_ref)
cumulative_averages_mapie = cumulative_average(empirical_coverages_mapie)

# Plot the results
fig, ax = plt.subplots()
plt.plot(cumulative_averages_ref, alpha=0.5, label='SplitCP', color='r')
plt.plot(cumulative_averages_mapie, alpha=0.5, label='MAPIE', color='g')

plt.hlines(exact_cov, 0, num_splits, color='r', ls='--', label='Exact Cov.')
plt.hlines(lower_bound, 0, num_splits, color='k', label='Lower Bound')
plt.hlines(upper_bound, 0, num_splits, color='b', label='Upper Bound')

plt.xlabel(r'Split Number')
plt.ylabel(r'$\overline{\mathbb{C}}$')
plt.title(r'$|D_{cal}| = $' + str(n_calib) + r' and $\delta = $' + str(delta))

plt.legend(loc="upper right", ncol=2)
plt.ylim(0.7, 1)
plt.tight_layout()
plt.show()


##############################################################################
# Section 2: Comparison with different MAPIE CP methods
# -----------------------------------------------------
#
# We propose to reproduce the previous experience with different methods of
# the MAPIE package (prefit, prefit with asymmetrical non-conformity scores
# and split).


def get_coverage_split(conformalizer, data, target, delta, random_state=None):
    """
    Calculate the fraction of test samples within the predicted intervals.

    This function splits the data into a training set and a test set. If the
    cross-validation strategy of the mapie regressor is a ShuffleSplit, it fits
    the regressor to the entire training set. Otherwise, it further splits the
    training set into a calibration set and a training set, and fits the
    regressor to the calibration set. It then predicts intervals for the test
    set and calculates the fraction of test samples within these intervals.

    Parameters:
    -----------
    conformalizer: object
        A mapie regressor object.

    data: array-like of shape (n_samples, n_features)
        The data to be split into a training set and a test set.

    target: array-like of shape (n_samples,)
        The target values for the data.

    delta: float
        The level of confidence for the predicted intervals.

    Returns:
    --------
    fraction_within_bounds: float
        The fraction of test samples within the predicted intervals.
    """
    # Split data step
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        data, target, test_size=n_test
    )

    # Calibration step
    if isinstance(conformalizer, MapieRegressor) and \
            isinstance(conformalizer.cv, ShuffleSplit):
        conformalizer.fit(X_train_cal, y_train_cal)
    else:
        _, X_cal, _, y_cal = train_test_split(
            X_train_cal, y_train_cal, test_size=n_calib
        )
        conformalizer.fit(X_cal, y_cal)

    # Prediction step
    if isinstance(conformalizer, StandardConformalizer):
        _, y_pis = conformalizer.predict(X_test)
    else:
        _, y_pis = conformalizer.predict(X_test, alpha=1-delta)

    # Coverage step
    fraction_within_bounds = regression_coverage_score_v2(y_test, y_pis)

    return fraction_within_bounds


def run_get_coverage_split(model, params, n_calib, data, target, delta):
    if not params:
        ref_reg = StandardConformalizer(model, non_conformity_func, delta)
        return get_coverage_split(ref_reg, data, target, delta)
    try:
        mapie_reg = MapieRegressor(estimator=model, **params(n_calib))
        coverage = get_coverage_split(mapie_reg, data, target, delta)
    except Exception:
        coverage = np.nan
    return coverage


STRATEGIES = {
    "reference": None,
    "prefit": lambda n: dict(
        method="base",
        cv="prefit",
        conformity_score=AbsoluteConformityScore(sym=True)
    ),
    "prefit_asym": lambda n: dict(
        method="base",
        cv="prefit",
        conformity_score=AbsoluteConformityScore(sym=False)
    ),
}


##############################################################################
# Experiment 3: Again but with different MAPIE CP methods
# -------------------------------------------------------
#
# The methods always follow different trajectories but always achieve the
# expected coverage.
# Since asymmetric scores can be used, the limits are not exactly the same.
# We should calculate them differently but that doesn't change our conclusion.

# Parameters of the modelisation
delta = 0.8
n_calib = 12  # for asymmetric non-conformity scores
num_splits = 1000

# Run the experiment
cumulative_averages_dict = dict()

for method, params in STRATEGIES.items():
    coverages_list = []
    run_params = model, params, n_calib, data, target, delta
    coverages_list = Parallel(n_jobs=-1)(
        delayed(run_get_coverage_split)(*run_params)
        for _ in range(num_splits)
    )
    cumulative_averages_dict[method] = cumulative_average(coverages_list)

# Plot the results
fig, ax = plt.subplots()
for method in STRATEGIES:
    plt.plot(cumulative_averages_dict[method], alpha=0.5, label=method)

plt.hlines(exact_cov, 0, num_splits, color='r', ls='--', label='Exact Cov.')
plt.hlines(lower_bound, 0, num_splits, color='k', label='Lower Bound')
plt.hlines(upper_bound, 0, num_splits, color='b', label='Upper Bound')

plt.xlabel(r'Split Number')
plt.ylabel(r'$\overline{\mathbb{C}}$')
plt.title(r'$|D_{cal}| = $' + str(n_calib) + r' and $\delta = $' + str(delta))

plt.legend(loc="upper right", ncol=2)
plt.ylim(0.7, 1)
plt.tight_layout()
plt.show()


##############################################################################
# Experiment 4: Extensive experimentation on different delta and n_calib
# ----------------------------------------------------------------------
#
# Here we propose to extend the experiment on different sizes of the
# calibration dataset and target coverage.
# We show the influence of size on effective coverage.
# In particular, we see that the expected coverage fluctuates between the
# limits with respect to the size of the calibration dataset but continues
# to converge towards the target coverage.
# It can be noted that all methods follow this trajectory and continue to
# achieve coverage validity.

num_splits = 100

nc_min, nc_max = 10, 30
n_calib_array = np.arange(nc_min, nc_max+1, 2)
delta = 0.8
delta_array = [delta]

final_coverage_dict = {
    method: {delta: [] for delta in delta_array}
    for method in STRATEGIES
}
effective_coverage_dict = {
    method: {delta: [] for delta in delta_array}
    for method in STRATEGIES
}

# Run experiment
for method, params in STRATEGIES.items():
    for n_calib in n_calib_array:
        coverages_list = []
        run_params = model, params, n_calib, data, target, delta
        coverages_list = Parallel(n_jobs=-1)(
            delayed(run_get_coverage_split)(*run_params)
            for _ in range(num_splits)
        )
        coverages_list = np.array(coverages_list)
        final_coverage = cumulative_average(coverages_list)[-1]
        final_coverage_dict[method][delta].append(final_coverage)


# Theorical bounds and exact coverage to attempt
def lower_bound_fct(delta):
    return delta * np.ones_like(n_calib_array)


def upper_bound_fct(delta):
    return delta + 1/(n_calib_array)


def upper_bound_asym_fct(delta):
    return delta + 1/(n_calib_array//2)


def exact_coverage_fct(delta):
    return np.ceil((n_calib_array+1)*delta)/(n_calib_array+1)


def exact_coverage_asym_fct(delta):
    new_n = n_calib_array//2-1
    return np.ceil((new_n+1)*delta)/(new_n+1)


# Plot the results
n_strat = len(final_coverage_dict)
nrows, ncols = n_strat, 1

fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

for i, method in enumerate(final_coverage_dict):
    # Compute the different bounds, target
    cov = final_coverage_dict[method][delta]
    ub = upper_bound_fct(delta)
    lb = lower_bound_fct(delta)
    exact_cov = exact_coverage_fct(delta)
    if 'asym' in method:
        ub = upper_bound_asym_fct(delta)
        exact_cov = exact_coverage_asym_fct(delta)
    ub = np.clip(ub, a_min=0, a_max=1)
    lb = np.clip(lb, a_min=0, a_max=1)

    # Plot the results
    ax[i].plot(n_calib_array, cov, alpha=0.5, label=method, color='g')
    ax[i].plot(n_calib_array, lb, color='k', label='Lower Bound')
    ax[i].plot(n_calib_array, ub, color='b', label='Upper Bound')
    ax[i].plot(n_calib_array, exact_cov, color='g', ls='--', label='Exact Cov')
    ax[i].hlines(delta, nc_min, nc_max, color='r', ls='--', label='Target Cov')

    ax[i].legend(loc="upper right", ncol=2)
    ax[i].set_ylim(np.min(lb) - 0.05, 1.0)
    ax[i].set_xlabel(r'$n_{calib}$')
    ax[i].set_ylabel(r'$\overline{\mathbb{C}}$')

fig.suptitle(r'$\delta = $' + str(delta))
plt.show()
