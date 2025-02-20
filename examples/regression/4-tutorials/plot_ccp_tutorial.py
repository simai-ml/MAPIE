"""
============================================
Tutorial: Conditional CP for regression
============================================

The tutorial will explain how to use the CCP method, and
will compare it with the other methods available in MAPIE. The CCP method
implements the method described in the Gibbs et al. (2023) paper [1].

We will see in this tutorial how to use the method. It has a lot of advantages:

- It is model agnostic (it doesn't depend on the model but only on the
  predictions, unlike `CQR`)
- It can create very adaptative intervals (with a varying width which truly
  reflects the model uncertainty)
- while providing coverage guarantee on all sub-groups of interest
  (avoiding biases)
- with the possibility to inject prior knowledge about the data or the model

However, we will also see its disadvantages:

- The adaptativity depends on the calibrator we use: It can be difficult to
  choose the correct calibrator,
  with the best parameters (this tutorial will try to help you with this task).
- The calibration and even more the inference are much longer than for the
  other methods. We can reduce the inference time using
  ``unsafe_approximation=True``, but we lose the strong theoretical guarantees
  and risk a small miscoverage
  (even if, most of the time, the coverage is achieved).

Conclusion on the method:

It can create more adaptative intervals than the other methods, but it can be
difficult to find the best settings (calibrator type and parameters)
and can have a big computational time.

----

In this tutorial, we will use a synthetic toy dataset.
The estimator will be :class:`~sklearn.pipeline.Pipeline`
with :class:`~sklearn.preprocessing.PolynomialFeatures` and
:class:`~sklearn.linear_model.LinearRegression` (or
:class:`~sklearn.linear_model.QuantileRegressor` for CQR).

We will compare the different available calibrators (
:class:`~mapie.future.calibrators.ccp.CustomCCP`,
:class:`~mapie.future.calibrators.ccp.GaussianCCP`
and :class:`~mapie.future.calibrators.ccp.PolynomialCCP`) of the CCP method
(using :class:`~mapie.future.split.SplitCPRegressor`), with the
standard split-conformal method, the CV+ method
(:class:`~mapie.regression.MapieRegressor`) and CQR
(:class:`~mapie.regression.MapieQuantileRegressor`)

Recall that the ``alpha`` is ``1 - target coverage``.

Warning:

In this tutorial, we use ``unsafe_approximation=True`` to have a faster
computation (because Read The Docs examples require fast computation).
This mode use an approximation, which make the inference (``predict``) faster,
but induce a small miscoverage. It is recommanded not to use it, or be
very careful and empirically check the coverage and a test set.

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees",
`arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.
"""

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.future.calibrators import CustomCCP, GaussianCCP, PolynomialCCP
from mapie.future.calibrators.ccp import CCPCalibrator
from mapie.future.split import SplitCPRegressor
from mapie.regression import MapieQuantileRegressor, MapieRegressor

warnings.filterwarnings("ignore")

random_state = 42
np.random.seed(random_state)

ALPHA = 0.1
UNSAFE_APPROXIMATION = True

##############################################################################
# 1. Data generation
# --------------------------------------------------------------------------
# Let's start by creating some synthetic data with different domains and
# distributions to evaluate the adaptativity of the methods:
#  - baseline distribution of ``x*sin(x)``
#  - Add noise :
#   - between -1 and 0: uniform distribution of the points around the baseline
#   - between 0 and 5: normal distribution with a noise value which
#     increase with ``x``
#
# We are going to use 5000 samples for training, 5000 for calibration and
# 5000 for testing.


def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x*np.sin(x)


def get_1d_data_with_heteroscedastic_noise(
    funct, min_x, max_x, n_samples, noise, power
):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    X = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X)
    y = (
        funct(X) +
        (np.random.normal(0, noise, len(X)) * ((X)/max_x)**power*max_x) +
        (np.random.uniform(-noise*3, noise*3, len(X))) * (X < 0)
    )
    true_pi = np.hstack([x_sinx(X).reshape(-1, 1)]*2)
    true_pi[X < 0, 0] += noise*3*(1-ALPHA)
    true_pi[X < 0, 1] -= noise*3*(1-ALPHA)
    true_pi[X >= 0, 0] += norm.ppf(1 - ALPHA/2) * noise * (
        ((X[X >= 0])/max_x)**power*max_x)
    true_pi[X >= 0, 1] -= norm.ppf(1 - ALPHA/2) * noise * (
        ((X[X >= 0])/max_x)**power*max_x)
    return X.reshape(-1, 1), y, true_pi


def generate_data(n_train=10000, n_test=5000, noise=0.8, power=2):
    X, y, true_pi = get_1d_data_with_heteroscedastic_noise(
        x_sinx, -1, 5, n_train + n_test, noise, power)
    indexes = list(range(len(X)))
    train_indexes = np.random.choice(indexes, n_train)
    indexes = list(set(indexes) - set(train_indexes))
    test_indexes = np.random.choice(indexes, n_test)
    return (X[train_indexes, :], y[train_indexes],
            X[test_indexes, :], y[test_indexes],
            true_pi[train_indexes, :], true_pi[test_indexes, :])


X_train, y_train, X_test, y_test, train_pi, test_pi = generate_data()


##############################################################################
# Let's visualize the data and its distribution

plt.scatter(X_train, y_train, color="C0", alpha=0.5, s=3,
            label="Training data")
sort_order = np.argsort(X_train[:, 0])
x_sorted = X_train[sort_order, :]
plt.plot(x_sorted, train_pi[sort_order, 0], "k--",
         label=f"True interval (alpha={ALPHA})")
plt.plot(x_sorted, train_pi[sort_order, 1], "k--", linestyle='--')
plt.plot(x_sorted, x_sinx(x_sorted), "k-", label="baseline")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data")
plt.legend()
plt.show()


##############################################################################
# 2. Model: Polynomial regression
# --------------------------------------------------------------------------

polynomial_degree = 4
quantile_estimator = Pipeline([
    ("poly", PolynomialFeatures(degree=polynomial_degree)),
    ("linear", QuantileRegressor(solver="highs", alpha=0))
])
estimator = Pipeline([
    ("poly", PolynomialFeatures(degree=polynomial_degree)),
    ("linear", LinearRegression())
])


##############################################################################
# 3. Plotting and adaptativity comparison functions
# --------------------------------------------------------------------------

def plot_subplot(ax, X, y, mapie, y_pred, upper_pi, lower_pi, color_rgb,
                 show_transform=False, ax_transform=None):
    """
    Plot the prediction interval and calibrator's features of a mapie instance
    """
    sort_order = np.argsort(X[:, 0])
    lw = 1
    color = mcolors.rgb2hex(color_rgb)
    x_test_sorted = X[sort_order]
    y_test_sorted = y[sort_order]
    y_pred_sorted = y_pred[sort_order]
    upper_pi_sorted = upper_pi[sort_order]
    lower_pi_sorted = lower_pi[sort_order]
    sample = np.random.choice(list(range(len(X))), min(4000, len(X)))
    # Plot test data
    ax.scatter(x_test_sorted[sample, 0], y_test_sorted[sample], s=1, alpha=0.3,
               color='darkblue', label="Test Data")
    # Plot prediction
    ax.plot(x_test_sorted[:, 0], y_pred_sorted, lw=lw,
            color='black', label="Prediction")
    # Plot prediction interval
    ax.fill_between(x_test_sorted[:, 0], upper_pi_sorted, lower_pi_sorted,
                    color=color, alpha=0.3, label="Prediction interval")
    # Plot upper and lower prediction intervals
    ax.plot(x_test_sorted[:, 0], upper_pi_sorted, lw=lw, color=color)
    ax.plot(x_test_sorted[:, 0], lower_pi_sorted, lw=lw, color=color)
    # Plot true prediction interval
    ax.plot(x_test_sorted[:, 0], test_pi[sort_order, 0], "--k",
            lw=lw*1.5, label='True Interval')
    ax.plot(x_test_sorted[:, 0], test_pi[sort_order, 1], "--k", lw=lw*1.5)

    if (
        show_transform and isinstance(mapie, SplitCPRegressor)
        and isinstance(mapie.calibrator_, CCPCalibrator)
    ):
        transform = mapie.calibrator_.transform(x_test_sorted)\
            * mapie.calibrator_.beta_up_[0]
        for i in range(transform.shape[1]):
            ax_transform.plot(
                x_test_sorted[:, 0],
                transform[:, i],
                lw=lw, color=color
            )


def has_ccp_calibrator(mapie):
    """
    Whether or not, the ``mapie`` instance has a ``CCPCalibrator`` calibrator
    """
    if (
        not isinstance(mapie, SplitCPRegressor)
        or not isinstance(mapie.calibrator_, CCPCalibrator)
    ):
        return False
    for calibrator in list(mapie.calibrator_.functions_) + [mapie.calibrator_]:
        if isinstance(calibrator, CCPCalibrator):
            return True
    return False


def plot_figure(mapies, y_preds, y_pis, titles, show_components=False):
    """
    Plot the prediction interval of mapie instances.
    Also plot the features of the calibrator, if ``show_transform=True``
    """
    cp = plt.get_cmap('tab10').colors
    ncols = min(3, len(titles))
    nrows = int(np.ceil(len(titles) / ncols))
    ax_need_transform = np.zeros((nrows, ncols))
    if show_components:
        for i, mapie in enumerate(mapies):
            ax_need_transform[i//ncols, i % ncols] = has_ccp_calibrator(mapie)
            row_need_transform = np.max(ax_need_transform, axis=1)
        height_ratio = np.array([
            item for x in row_need_transform
            for item in ([3] if x == 0 else [3, 1])
        ])
        fig, axes = plt.subplots(
            nrows=nrows + int(sum(row_need_transform)), ncols=ncols,
            figsize=(ncols*3.6, nrows*3.6 + int(sum(row_need_transform))*1.8),
            height_ratios=height_ratio
        )

        transform_axes = np.full((nrows, ncols), None)
        transform_axes[row_need_transform == 1, :] = axes[height_ratio == 1, :]
        transform_axes = transform_axes.flatten()
        main_axes = axes[height_ratio == 3, :].flatten()
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(ncols*4, nrows*4))
        main_axes = axes.flatten()
        transform_axes = np.full(main_axes.shape, None)

    for i in range(len(mapies), len(main_axes)):
        fig.delaxes(main_axes[i])
        if transform_axes[i] is not None:
            fig.delaxes(transform_axes[i])

    for i, (m_ax, t_ax, mapie, y_pred, y_pi, title) in enumerate(
        zip(main_axes, transform_axes, mapies, y_preds, y_pis, titles)
    ):
        lower_bound = y_pi[:, 0, 0]
        upper_bound = y_pi[:, 1, 0]

        plot_subplot(
            m_ax, X_test, y_test, mapie, y_pred, upper_bound, lower_bound,
            cp[i], show_transform=ax_need_transform.flatten()[i],
            ax_transform=t_ax
        )
        m_ax.set_title(title)
        if i % 3 == 0:
            m_ax.set_ylabel('Y')
        if t_ax is not None:
            t_ax.set_title("Components of the PI")
            if i >= len(titles) - ncols:
                t_ax.set_xlabel('X')
            if i % 3 == 0:
                t_ax.set_ylabel('component value')
        m_ax.set_xlabel('X')
        m_ax.legend()

    fig.tight_layout()
    plt.show()


def compute_conditional_coverage(X_test, y_test, y_pis, bins_width=0.25):
    """
    Compute the conditional coverage on ``X_test``, using discret bins
    """
    bin_edges = np.arange(np.min(X_test), np.max(X_test) + bins_width,
                          bins_width)
    coverage = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_edges) - 1):
        in_bin = np.logical_and(X_test[:, 0] >= bin_edges[i],
                                X_test[:, 0] < bin_edges[i + 1])
        coverage[i] = np.mean(np.logical_and(
            y_test[in_bin] >= y_pis[in_bin, 0, 0],
            y_test[in_bin] <= y_pis[in_bin, 1, 0]
        ))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, coverage


def plot_evaluation(titles, y_pis, X_test, y_test):
    """
    Plot the conditional coverages
    """
    sort_order = np.argsort(X_test[:, 0])
    cp = plt.get_cmap('tab10').colors

    num_plots = len(titles)
    num_rows = (num_plots + 2) // 3

    fig, axs = plt.subplots(nrows=num_rows, ncols=2,
                            figsize=(10, 3.7*num_rows))
    if len(axs.shape) == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()  # Flatten to make indexing easier

    cov_lim = [1, 0]
    width_lim = [np.inf, 0]
    for i in range(num_rows):
        for j, pi in enumerate(y_pis[3*i: 3*(i+1)]):
            c = mcolors.rgb2hex(cp[i*3+j])
            # Conditionnal coverage
            bin_centers, coverage = compute_conditional_coverage(
                X_test, y_test, pi
            )
            axs[i * 2].plot(bin_centers, coverage, lw=2, color=c)
            axs[i * 2].axhline(
                y=np.mean(coverage), color=c, linestyle="--",
                label=f"Coverage={round(np.mean(coverage)*100, 1)}%"
            )
            axs[i * 2].axhline(
                y=1-ALPHA, color='black', linestyle="--",
                label=(
                    f"alpha={ALPHA}" if j == len(y_pis[3*i: 3*(i+1)]) - 1
                    else None
                )
            )
            cov_lim[0] = min(cov_lim[0], min(coverage))
            cov_lim[1] = max(cov_lim[1], max(coverage))
            # Interval width
            width = pi[sort_order, 1, 0] - pi[sort_order, 0, 0]
            axs[i * 2 + 1].plot(
                X_test[sort_order, 0],
                width,
                lw=2, color=c, label=titles[i*3+j]
            )

            width_lim[0] = min(width_lim[0], min(width))
            width_lim[1] = max(width_lim[1], max(width))
        perfect_width = test_pi[sort_order, 0] - test_pi[sort_order, 1]
        axs[i * 2 + 1].plot(
            X_test[sort_order, 0],
            perfect_width,
            lw=2, color='black', linestyle="--", label="Perfect Width"
        )
        width_lim[0] = min(width_lim[0], min(perfect_width))
        width_lim[1] = max(width_lim[1], max(perfect_width))

        axs[i * 2 + 1].legend(fontsize=10)
        axs[i * 2 + 1].set_title("Prediction Interval Width")
        axs[i * 2 + 1].set_xlabel("X")
        axs[i * 2 + 1].set_ylabel("Width")
        axs[i * 2].legend(fontsize=10)
        axs[i * 2].set_title("Conditional Coverage")
        axs[i * 2].set_xlabel("X (bins of 0.5 width)")
        axs[i * 2].set_ylabel("Coverage")

    # Remove unused subplots
    for j in range(num_plots * 2, len(axs)):
        fig.delaxes(axs[j])

    for ax_cov, ax_width in zip(axs[::2], axs[1::2]):
        ax_cov.set_ylim([cov_lim[0]*0.95, cov_lim[1]*1.05])
        ax_width.set_ylim([width_lim[0]*0.95, width_lim[1]*1.05])

    plt.tight_layout()
    plt.show()


##############################################################################
# 4. Creation of Mapie instances
# --------------------------------------------------------------------------
# We are going to test different methods : ``CV+``, ``CQR`` and ``CCP``
# (with default parameters)

cv = ShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)

# ================== Basic Split-conformal  ==================
mapie_split = MapieRegressor(estimator, method="base", cv=cv)
mapie_split.fit(X_train, y_train)
y_pred_split, y_pi_split = mapie_split.predict(X_test, alpha=ALPHA)

# ================== CV+  ==================
# MapieRegressor defaults to method='plus' and cv=5
mapie_cv = MapieRegressor(estimator)
mapie_cv.fit(X_train, y_train)
y_pred_cv, y_pi_cv = mapie_cv.predict(X_test, alpha=ALPHA)

# ================== CQR  ==================
mapie_cqr = MapieQuantileRegressor(quantile_estimator, alpha=ALPHA)
mapie_cqr.fit(X_train, y_train)
y_pred_cqr, y_pi_cqr = mapie_cqr.predict(X_test)

# ================== CCP  ==================
# `SplitCPRegressor` defaults to `calibrator=GaussianCCP()``
mapie_ccp = SplitCPRegressor(estimator, calibrator=GaussianCCP(),
                             alpha=ALPHA, cv=cv)
mapie_ccp.fit(X_train, y_train)
y_pred_ccp, y_pi_ccp = mapie_ccp.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

# ================== PLOT ==================
mapies = [mapie_split, mapie_cv, mapie_cqr, mapie_ccp]
y_preds = [y_pred_split, y_pred_cv, y_pred_cqr, y_pred_ccp]
y_pis = [y_pi_split, y_pi_cv, y_pi_cqr, y_pi_ccp]
titles = ["Basic Split", "CV+", "CQR", "CCP (default)"]

plot_figure(mapies, y_preds, y_pis, titles)
plot_evaluation(titles, y_pis, X_test, y_test)


##############################################################################
# The :class:`~mapie.future.split.regression.SplitCPRegressor` has is
# a very adaptative method, even with default
# parameters values. If the dataset is more complex, the default parameters
# may not be enough to get the best performances. In this case, we can use
# more advanced settings, described below.


##############################################################################
# 5. How to improve the results?
# --------------------------------------------------------------------------
#
# 5.1. How does the ``CCP`` method works ?
# --------------------------------------------------------------------------
# The CCP method is based on a function which create some features(vector of
# d dimensions), based on ``X`` (and potentially the prediction ``y_pred``).
#
# These features should be able to represente the distribuion of the
# conformity scores, which is here (by default) the absolute residual:
# ``|y_true - y_pred|``

##############################################################################
# Examples of basic functions:
# --------------------------------------------------------------------------
#
##############################################################################

##############################################################################
#  1) ``f : X -> (1)``, will try to estimate the absolute residual with a
#  constant, and will results in a prediction interval of constant width
#  (like the basic split CP)
#
#  2) ``f : X -> (1, X)``, will result in a prediction interval of width
#  equal to: a constant + a value proportional to the value of ``X``
#  (it seems a good idea here, as the uncertainty increase with ``X``)
#
#  3) ``f : X, y_pred -> (y_pred)``, will result in a prediction interval
#  of width proportional to the prediction (Like the basic split CP with a
#  gamma conformity score).


##############################################################################
# Using custom definition
# --------------------------------------------------------------------------
#
##############################################################################


calibrator1 = CustomCCP([lambda X: np.ones(len(X))])
calibrator1_bis = CustomCCP(bias=True)
# calibrator1_bis is equivalent to calibrator1,
# as bias=True adds a column of ones
calibrator2 = CustomCCP([lambda X: X], bias=True)
calibrator3 = CustomCCP([lambda y_pred: y_pred])


##############################################################################
# Or using :class:`~mapie.future.calibrators.ccp.PolynomialCCP` class:
# --------------------------------------------------------------------------
#
##############################################################################

calibrator1 = PolynomialCCP(0)
calibrator2 = PolynomialCCP(1)  # degree=1 is equivalent to degree=[0, 1]
calibrator3 = PolynomialCCP([1], variable="y_pred")
# Note: adding '0' in the 'degree' argument list
# is equivalent tohaving bias=True, as X^0=1


##############################################################################
# 5.2. Improve the performances without prior knowledge: :class:`GaussianCCP`
# --------------------------------------------------------------------------
# If we don't know anything about the data, we can use
# :class:`~mapie.future.calibrators.ccp.GaussianCCP`,
# which will sample random points, and apply gaussian kernels
# with a given standard deviation ``sigma``.
#
# Basically, the conformity score of a given point ``x_test``,
# will be estimated based on the conformity scores
# of calibration samples which are closed to ``x_test``.
# It result in a globally good adaptativity.
#
# The ``sigma`` hyperparameter can be optimized using cross-validation.
# It is defined by default based on the standard deviaiton of ``X``.

calibrator_gauss1 = GaussianCCP(np.arange(-1, 6).reshape(-1, 1), 1)
calibrator_gauss2 = GaussianCCP(30, 0.05)
calibrator_gauss3 = GaussianCCP(30, 0.25, random_sigma=True)

# # ================== CCP 1  ==================
mapie_ccp_1 = SplitCPRegressor(estimator, calibrator=calibrator_gauss1,
                               cv=cv, alpha=ALPHA)
mapie_ccp_1.fit(X_train, y_train)
y_pred_ccp_1, y_pi_ccp_1 = mapie_ccp_1.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

# # ================== CCP 2 ==================
mapie_ccp_2 = SplitCPRegressor(estimator, calibrator=calibrator_gauss2,
                               cv=cv, alpha=ALPHA)
mapie_ccp_2.fit(X_train, y_train)
y_pred_ccp_2, y_pi_ccp_2 = mapie_ccp_2.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

# # ================== CCP 3  ==================
mapie_ccp_3 = SplitCPRegressor(estimator, calibrator=calibrator_gauss3,
                               cv=cv, alpha=ALPHA)
mapie_ccp_3.fit(X_train, y_train)
y_pred_ccp_3, y_pi_ccp_3 = mapie_ccp_3.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)


mapies = [mapie_split, mapie_cv, mapie_cqr,
          mapie_ccp_1, mapie_ccp_2, mapie_ccp_3]
y_preds = [y_pred_split, y_pred_cv, y_pred_cqr,
           y_pred_ccp_1, y_pred_ccp_2, y_pred_ccp_3]
y_pis = [y_pi_split, y_pi_cv, y_pi_cqr,
         y_pi_ccp_1, y_pi_ccp_2, y_pi_ccp_3]
titles = ["Basic Split", "CV+", "CQR",
          "CCP 1: 6 points, s=1 (under-fit)",
          "CCP 2: 30 points, s=0.05 (over-fit)",
          "CCP 3: 30 points, s=0.25 (good calibrator)"]

plot_figure(mapies, y_preds, y_pis, titles, show_components=True)
plot_evaluation(titles, y_pis, X_test, y_test)

##############################################################################
# --> Using gaussian distances (with correct sigma value) from randomly
# sampled points is a good solution to have an overall good adaptativity.

##############################################################################
# 5.3. Improve the performances using what we know about the data
# --------------------------------------------------------------------------
# To improve the results, we need to analyse the data
# and the conformity scores we chose (here, the absolute residuals).
#
#  1) We can see that the residuals (error with the prediction)
#  increase with X, for X > 0.
#
#  2) For X < 0, the points seem uniformly distributed around
#  the base distribution.
#
# --> It should be a good idea to inject in the calibrator the two groups
# ( X < 0 and X > 0). We can use on each group
# :class:`~mapie.future.calibrators.ccp.GaussianCCP`
# (or :class:`~mapie.future.calibrators.ccp.PolynomialCCP`,
# as it seems adapted in this example)

calibrator1 = CustomCCP(
    [lambda X: X < 0, (lambda X: X >= 0)*PolynomialCCP(3)]
)
calibrator2 = CustomCCP(
    [
        (lambda X: X < 0)*PolynomialCCP(3),
        (lambda X: X >= 0)*PolynomialCCP(3)
    ]
)
calibrator3 = CustomCCP(
    [
        (lambda X: X < 0)*GaussianCCP(5),
        (lambda X: X >= 0)*GaussianCCP(30)
    ],
    normalized=True,
)

# ================== CCP 1  ==================
mapie_ccp_1 = SplitCPRegressor(estimator, calibrator=calibrator1,
                               cv=cv,  alpha=ALPHA)
mapie_ccp_1.fit(X_train, y_train)
y_pred_ccp_1, y_pi_ccp_1 = mapie_ccp_1.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

# ================== CCP 2  ==================
mapie_ccp_2 = SplitCPRegressor(estimator, calibrator=calibrator2,
                               cv=cv, alpha=ALPHA)
mapie_ccp_2.fit(X_train, y_train)
y_pred_ccp_2, y_pi_ccp_2 = mapie_ccp_2.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

# ================== CCP 3  ==================
mapie_ccp_3 = SplitCPRegressor(estimator, calibrator=calibrator3,
                               cv=cv, alpha=ALPHA)
mapie_ccp_3.fit(X_train, y_train)
y_pred_ccp_3, y_pi_ccp_3 = mapie_ccp_3.predict(
    X_test, unsafe_approximation=UNSAFE_APPROXIMATION
)

mapies = [mapie_split, mapie_cv, mapie_cqr,
          mapie_ccp_1, mapie_ccp_2, mapie_ccp_3]
y_preds = [y_pred_split, y_pred_cv, y_pred_cqr,
           y_pred_ccp_1, y_pred_ccp_2, y_pred_ccp_3]
y_pis = [y_pi_split, y_pi_cv, y_pi_cqr,
         y_pi_ccp_1, y_pi_ccp_2, y_pi_ccp_3]
titles = ["Basic Split", "CV+", "CQR",
          "CCP 1: const (X<0) / poly (X>0)",
          "CCP 2: poly (X<0) / poly (X>0)",
          "CCP: gauss (X<0) / gauss (X>0)"]


plot_figure(mapies, y_preds, y_pis, titles, show_components=True)
plot_evaluation(titles, y_pis, X_test, y_test)

##############################################################################
# 6. Conclusion:
# --------------------------------------------------------------------------
# The goal is to get prediction intervals which are the most adaptative
# possible. Perfect adaptativity whould result in a perfectly constant
# conditional coverage.
#
# Considering this adaptativity criteria, the most adaptative interval is
# this last brown one, with the two groups
# and the gaussian calibrators. In this example, the polynomial
# calibrator (in purple) also worked well, but the gaussian one is more generic
# (It usually work with any dataset, assuming we use the correct parameters,
# whereas the polynomial features are not always adapted).
#
# This is the power of the ``CCP`` method: combining prior knowledge and
# generic features (gaussian kernelsl) to have a great overall adaptativity.
#
# However, it can be difficult to find the best calibrator and parameters.
# Sometimes, a simpler method (standard ``split`` with ``GammaConformityScore``
# for example) can be enough. Don't forget to try at first the simpler method,
# and move on with the more advanced if it is necessary.
