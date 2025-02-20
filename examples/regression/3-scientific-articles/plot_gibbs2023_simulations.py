"""
======================================================================
Reproduction of part of the paper experiments of Gibbs et al. (2023)
======================================================================

:class:`~mapie.regression.MapieCCPRegressor` is used to reproduce a
part of the paper experiments of Gibbs et al. (2023) in their article [1]
which we argue is a good procedure to get adaptative prediction intervals (PI)
and a guaranteed coverage on all sub groups of interest.

For a given model, the simulation adjusts the MAPIE regressors using the
``CCP`` method, on a synthetic dataset first considered by Romano et al. (2019)
[2], and compares the bounds of the PIs with the standard split CP.

In order to reproduce the results of the standard split conformal prediction
(Split CP), we reuse the Mapie implementation in
:class:`~mapie.regression.MapieRegressor`.

This simulation is carried out to check that the CCP method implemented in
MAPIE gives the same results as [1], and that the bounds of the PIs are
obtained.

It is important to note that we are checking here if the adaptativity property
of the prediction intervals are well obtained. However, the paper do this
computations with the full conformal prediction approach, whereas we
implemented the faster but more conservatice split method. Thus, the results
may vary a little.

[1] Isaac Gibbs, John J. Cherian, Emmanuel J. Candès (2023).
Conformal Prediction With Conditional Guarantees

[2] Yaniv Romano, Evan Patterson, Emmanuel J. Candès (2019).
Conformalized Quantile Regression.
33rd Conference on Neural Information Processing Systems (NeurIPS 2019).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.future.calibrators.ccp import CustomCCP, GaussianCCP
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import MapieRegressor
from mapie.future.split import SplitCPRegressor

warnings.filterwarnings("ignore")

random_state = 1
np.random.seed(random_state)


###############################################################################
# 1. Global model parameters
# -----------------------------------------------------------------------------

def init_model():
    # the degree of the polynomial regression
    degree = 4

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("linear", LinearRegression())
        ]
    )
    return model

###############################################################################
# 2. Generate and present data
# -----------------------------------------------------------------------------


def generate_data(n_train=2000, n_calib=2000, n_test=500):
    def f(x):
        ax = 0*x
        for i in range(len(x)):
            ax[i] = (np.random.poisson(np.sin(x[i])**2 + 0.1)
                     + 0.03*x[i]*np.random.randn(1))
            ax[i] += 25*(np.random.uniform(0, 1, 1) < 0.01)*np.random.randn(1)
        return ax.astype(np.float32)

    # training features
    X_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)
    X_calib = np.random.uniform(0, 5.0, size=n_calib).astype(np.float32)
    X_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)

    # generate labels
    y_train = f(X_train)
    y_calib = f(X_calib)
    y_test = f(X_test)

    # reshape the features
    X_train = X_train.reshape(-1, 1)
    X_calib = X_calib.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    return X_train, y_train, X_calib, y_calib, X_test, y_test


X_train, y_train, X_calib, y_calib, X_test, y_test = generate_data()

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_train[:, 0], y_train, s=1.5, alpha=0.6, label="Train Data")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Train Data")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(X_train[:, 0], y_train, s=1.5, alpha=0.6, label="Train Data")
ax2.set_ylim([-2, 6])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Zoom")
ax2.legend()

plt.show()

##############################################################################
# 3. Prepare model and show predictions
# -----------------------------------------------------------------------------

model = init_model()

model.fit(X_train, y_train)

sort_order = np.argsort(X_test[:, 0])
x_test_s = X_test[sort_order]
y_pred_s = model.predict(x_test_s)

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], y_test, s=1.5, alpha=0.6, label="Test Data")
plt.plot(x_test_s, y_pred_s, "-k", label="Prediction")
plt.ylim([-2, 6])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Test Data (Zoom)")
plt.legend()
plt.show()


##############################################################################
# 4. Prepare Experiments
# -----------------------------------------------------------------------------
# In this experiment, we will use the
# :class:`~mapie.regression.MapieRegressor` and
# :class:`~mapie.regression.MapieCCPRegressor` to compute prediction intervals
# with the basic Split CP method and the paper CCP method.
# The coverages was computed, in the paper, on 500 different dataset
# generations, to have a good idea of the true value.
# Indeed, the empirical coverage of a single
# experiment is stochastic, because of the finite number of calibration and
# test samples.
# We will only compute 50 trials, because of the documentation
# computational power limitations.

ALPHA = 0.1


def estimate_coverage(mapie_split, mapie_ccp, group_functs=[]):
    _, _, X_calib, y_calib, X_test, y_test = generate_data()

    mapie_split.fit(X_calib, y_calib)
    _, y_pi_split = mapie_split.predict(X_test, alpha=ALPHA)

    mapie_ccp.fit_calibrator(X_calib, y_calib)
    _, y_pi_ccp = mapie_ccp.predict(X_test)

    cover_split = np.logical_or(y_test < y_pi_split[:, 0, 0],
                                y_test > y_pi_split[:, 1, 0])
    cover_ccp = np.logical_or(y_test < y_pi_ccp[:, 0, 0],
                              y_test > y_pi_ccp[:, 1, 0])
    group_covers = []
    marginal_cover = np.asarray((cover_split.mean(), cover_ccp.mean()))
    for funct in group_functs:
        group_cover = np.zeros((2,))
        group_cover[0] = (funct(X_test).flatten()
                          * cover_split).sum() / funct(X_test).sum()
        group_cover[1] = (funct(X_test).flatten()
                          * cover_ccp).sum() / funct(X_test).sum()
        group_covers.append(group_cover)
    return marginal_cover, np.array(group_covers)


def plot_results(X_test, y_test, n_trials=10,
                 experiment="Groups", split_sym=True):

    # Split CP
    mapie_split = MapieRegressor(
        model, method="base", cv="prefit",
        conformity_score=AbsoluteConformityScore(sym=split_sym)
    )
    mapie_split.conformity_score.eps = 1e-5
    mapie_split.fit(X_calib, y_calib)
    _, y_pi_split = mapie_split.predict(X_test, alpha=ALPHA)

    if experiment == "Groups":
        # CCP Groups
        calibrator_groups = CustomCCP([
            lambda X, t=t: np.logical_and(X >= t, X < t + 0.5).astype(int)
            for t in np.arange(0, 5.5, 0.5)
        ])
        mapie_ccp = SplitCPRegressor(
            model, calibrator=calibrator_groups, alpha=ALPHA, cv="prefit",
            conformity_score=AbsoluteConformityScore(sym=False),
            random_state=None
        )
        mapie_ccp.conformity_score.eps = 1e-5
        mapie_ccp.fit(X_calib, y_calib)
        _, y_pi_ccp = mapie_ccp.predict(X_test)
    else:
        # CCP Shifts
        eval_locs = [1.5, 3.5]
        eval_scale = 0.2
        other_locs = [0.5, 2.5, 4.5]
        other_scale = 1

        calibrator_shifts = GaussianCCP(
            points=(
                np.array(eval_locs+other_locs).reshape(-1, 1),
                [eval_scale]*len(eval_locs) + [other_scale]*len(other_locs),
            ),
            bias=True,
            normalized=False,
        )
        mapie_ccp = SplitCPRegressor(
            model, calibrator=calibrator_shifts, alpha=ALPHA, cv="prefit",
            conformity_score=AbsoluteConformityScore(sym=False),
            random_state=None
        )
        mapie_ccp.conformity_score.eps = 1e-5
        mapie_ccp.fit(X_calib, y_calib)
        _, y_pi_ccp = mapie_ccp.predict(X_test)

    # =========== n_trials run to get average marginal coverage ============
    if experiment == "Groups":
        eval_functions = [
            lambda X, a=a, b=b: np.logical_and(X >= a, X <= b).astype(int)
            for a, b in zip([1, 3], [2, 4])
        ]
        eval_names = ["[1, 2]", "[3, 4]"]
    else:
        eval_functions = [
            lambda x: norm.pdf(x, loc=1.5, scale=0.2).reshape(-1, 1),
            lambda x: norm.pdf(x, loc=3.5, scale=0.2).reshape(-1, 1)
        ]
        eval_names = ["f1", "f2"]

    marginal_cov = np.zeros((n_trials, 2))
    group_cov = np.zeros((len(eval_functions), n_trials, 2))
    for j in range(n_trials):
        marginal_cov[j], group_cov[:, j, :] = estimate_coverage(
            mapie_split, mapie_ccp, eval_functions
        )

    coverageData = pd.DataFrame()

    for group, cov in zip(["Marginal"]+eval_names,
                          [marginal_cov] + list(group_cov)):
        for i, name in enumerate(["Split", "CCP"]):
            coverageData = pd.concat(
                [coverageData,
                 pd.DataFrame({'Method': [name] * len(cov),
                               'Range': [group] * len(cov),
                               'Miscoverage': np.asarray(cov)[:, i]})],
                axis=0
            )

    # ================== results plotting ==================
    cp = plt.get_cmap('tab10').colors

    # Set font and style
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.grid'] = False

    fig = plt.figure()
    fig.set_size_inches(17, 6)

    sort_order = np.argsort(X_test[:, 0])
    x_test_s = X_test[sort_order]
    y_test_s = y_test[sort_order]
    y_pred_s = model.predict(x_test_s)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x_test_s, y_test_s, '.', alpha=0.2)
    ax1.plot(x_test_s, y_pred_s, lw=1, color='k')
    ax1.plot(x_test_s, y_pi_split[sort_order, 0, 0], color=cp[0], lw=2)
    ax1.plot(x_test_s, y_pi_split[sort_order, 1, 0], color=cp[0], lw=2)
    ax1.fill_between(x_test_s.flatten(), y_pi_split[sort_order, 0, 0],
                     y_pi_split[sort_order, 1, 0],
                     color=cp[0], alpha=0.4, label='split prediction interval')
    ax1.set_ylim(-2, 6.5)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xlabel("$X$", fontsize=16, labelpad=10)
    ax1.set_ylabel("$Y$", fontsize=16, labelpad=10)
    ax1.set_title("Split calibration", fontsize=18, pad=12)

    if experiment == 'Groups':
        ax1.axvspan(1, 2, facecolor='grey', alpha=0.25)
        ax1.axvspan(3, 4, facecolor='grey', alpha=0.25)
    else:
        for loc in eval_locs:
            ax1.plot(x_test_s, norm.pdf(x_test_s, loc=loc, scale=eval_scale),
                     color='grey', ls='--', lw=3)

    ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax2.plot(x_test_s, y_test_s, '.', alpha=0.2)
    ax2.plot(x_test_s, y_pred_s, color='k', lw=1)
    ax2.plot(x_test_s, y_pi_ccp[sort_order, 0, 0], color=cp[1], lw=2)
    ax2.plot(x_test_s, y_pi_ccp[sort_order, 1, 0], color=cp[1], lw=2)
    ax2.fill_between(x_test_s.flatten(), y_pi_ccp[sort_order, 0, 0],
                     y_pi_ccp[sort_order, 1, 0], color=cp[1], alpha=0.4,
                     label='conditional calibration')
    ax2.tick_params(axis='both', which='major', direction='out', labelsize=14)
    ax2.set_xlabel("$X$", fontsize=16, labelpad=10)
    ax2.set_ylabel("$Y$", fontsize=16, labelpad=10)
    ax2.set_title("Conditional calibration", fontsize=18, pad=12)

    if experiment == 'Groups':
        ax2.axvspan(1, 2, facecolor='grey', alpha=0.25)
        ax2.axvspan(3, 4, facecolor='grey', alpha=0.25)
    else:
        for loc in eval_locs:
            ax2.plot(x_test_s, norm.pdf(x_test_s, loc=loc, scale=eval_scale),
                     color='grey', ls='--', lw=3)

    ax3 = fig.add_subplot(1, 3, 3)

    ranges = coverageData['Range'].unique()
    methods = coverageData['Method'].unique()
    bar_width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        method_data = coverageData[coverageData['Method'] == method]
        x = np.arange(len(ranges)) + i * bar_width
        ax3.bar(x, method_data.groupby("Range")['Miscoverage'].mean(),
                width=bar_width, label=method, color=cp[i])

    ax3.set_xticks(np.arange(len(ranges)) + bar_width * (len(methods) - 1) / 2)
    ax3.set_xticklabels(ranges)

    ax3.axhline(0.1, color='red')
    ax3.legend()
    ax3.set_ylabel("Miscoverage", fontsize=18, labelpad=10)
    ax3.set_xlabel(experiment, fontsize=18, labelpad=10)
    ax3.set_ylim(0., 0.2)
    ax3.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout(pad=2)
    plt.show()


##############################################################################
# 5. Reproduce experiment and results
# -----------------------------------------------------------------------------

plot_results(X_test, y_test, 20, experiment="Groups")

plot_results(X_test, y_test, 20, experiment="Shifts")


##############################################################################
# We succesfully reproduced the experiement of the Gibbs et al. paper [1].
