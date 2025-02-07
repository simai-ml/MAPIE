"""
============================================
Tutorial: Conditional CP for classification
============================================

The tutorial will explain how to use the CCP method for classification
and will wompare it with the other methods available in MAPIE. The CCP method
implements the method described in the Gibbs et al. (2023) paper [1].

In this tutorial, the classifier will be
:class:`~sklearn.linear_model.LogisticRegression`.
We will use a synthetic toy dataset.

We will compare the CCP method (using
:class:`~mapie.future.split.SplitCPRegressor`,
:class:`~mapie.future.calibrators.ccp.CustomCCP` and
:class:`~mapie.future.calibrators.ccp.GaussianCCP`), with the
standard method, using for both, the LAC conformity score
(:class:`~mapie.conformity_scores.LACConformityScore`).

Recall that the ``LAC`` method consists on applying a threshold on the
predicted softmax, to keep all the classes above the threshold
(``alpha`` is ``1 - target coverage``).

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression

from mapie.future.calibrators import CustomCCP, GaussianCCP
from mapie.classification import MapieClassifier
from mapie.conformity_scores import LACConformityScore
from mapie.future.split.classification import SplitCPClassifier

warnings.filterwarnings("ignore")

random_state = 1
np.random.seed(random_state)

ALPHA = 0.2
UNSAFE_APPROXIMATION = True
N_CLASSES = 5

##############################################################################
# 1. Data generation
# --------------------------------------------------------------------------
# Let's start by creating some synthetic data with 5 gaussian distributions
#
# We are going to use 5000 samples for training, 3000 for calibration and
# 10000 for testing (to have a good conditional coverage evaluation).


def create_toy_dataset(n_samples=1000):
    centers = [(0, 3.5), (-3, 0), (0, -2), (4, -1), (3, 1)]
    covs = [
        np.diag([1, 1]), np.diag([2, 2]), np.diag([3, 2]),
        np.diag([3, 3]), np.diag([2, 2]),
    ]
    n_per_class = (
        np.linspace(0, n_samples, N_CLASSES + 1)[1:]
        - np.linspace(0, n_samples, N_CLASSES + 1)[: -1].astype(int)
    ).astype(int)
    X = np.vstack([
        np.random.multivariate_normal(center, cov, n)
        for center, cov, n in zip(centers, covs, n_per_class)

    ])
    y = np.hstack([np.full(n_per_class[i], i) for i in range(N_CLASSES)])

    return X, y


def generate_data(seed=1, n_train=2000, n_calib=2000, n_test=2000, ):
    np.random.seed(seed)
    x_train, y_train = create_toy_dataset(n_train)
    x_calib, y_calib = create_toy_dataset(n_calib)
    x_test, y_test = create_toy_dataset(n_test)

    return x_train, y_train, x_calib, y_calib, x_test, y_test

##############################################################################
# Let's visualize the data and its distribution


x_train, y_train, *_ = generate_data(seed=None, n_train=1000)

for c in range(N_CLASSES):
    plt.scatter(x_train[y_train == c, 0], x_train[y_train == c, 1],
                c=f"C{c}", s=1.5, label=f'Class {c}')
plt.legend()
plt.show()


##############################################################################
# 2. Plotting and adaptativity comparison functions
# --------------------------------------------------------------------------


def run_exp(
    mapies, names, alpha,
    n_train=1000, n_calib=1000, n_test=1000,
    grid_step=100, plot=True, seed=1, max_display=2000
):
    (
        x_train, y_train, x_calib, y_calib, x_test, y_test
    ) = generate_data(
        seed=seed, n_train=n_train, n_calib=n_calib, n_test=n_test
    )

    if max_display:
        display_ind = np.random.choice(np.arange(0, len(x_test)), max_display)
    else:
        display_ind = np.arange(0, len(x_test))

    color_map = plt.cm.get_cmap("Purples", N_CLASSES + 1)

    if plot:
        fig = plt.figure()
        fig.set_size_inches(6 * (len(mapies) + 1), 7)
        grid = plt.GridSpec(1, len(mapies) + 1)

        x_min = np.min(x_train)
        x_max = np.max(x_train)
        step = (x_max - x_min) / grid_step

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step), np.arange(x_min, x_max, step)
        )
        X_test_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

    scores = np.zeros((len(mapies), N_CLASSES+1))
    for i, (mapie, name) in enumerate(zip(mapies, names)):
        if isinstance(mapie, MapieClassifier):
            mapie.fit(
                np.vstack([x_train, x_calib]), np.hstack([y_train, y_calib])
            )
            _, y_ps_test = mapie.predict(x_test, alpha=alpha)
            if plot:
                y_pred_mesh, y_ps_mesh = mapie.predict(
                    X_test_mesh, alpha=alpha
                )
        elif isinstance(mapie, SplitCPClassifier):
            mapie.fit(
                np.vstack([x_train, x_calib]), np.hstack([y_train, y_calib])
            )
            _, y_ps_test = mapie.predict(
                x_test, unsafe_approximation=UNSAFE_APPROXIMATION
            )
            if plot:
                y_pred_mesh, y_ps_mesh = mapie.predict(X_test_mesh)
        else:
            raise

        if plot:
            if i == 0:
                ax1 = fig.add_subplot(grid[0, 0])

                ax1.scatter(
                    X_test_mesh[:, 0], X_test_mesh[:, 1],
                    c=[f"C{x}" for x in y_pred_mesh], alpha=1, marker="s",
                    edgecolor="none", s=220 * step
                )
                ax1.fill_between(
                    x=[min(X_test_mesh[:, 0]) - step] + list(X_test_mesh[:, 0])
                    + [max(X_test_mesh[:, 0]) + step],
                    y1=min(X_test_mesh[:, 1]) - step,
                    y2=max(X_test_mesh[:, 1]) + step,
                    color="white", alpha=0.6
                )
                ax1.scatter(
                    x_test[display_ind, 0], x_test[display_ind, 1],
                    c=[f"C{x}" for x in y_test[display_ind]],
                    alpha=1, marker=".", edgecolor="black", s=80
                )

                ax1.set_title("Predictions", fontsize=22, pad=12)
                ax1.set_xlim([-6, 8])
                ax1.set_ylim([-6, 8])
                legend_labels = [f"Class {i}" for i in range(N_CLASSES)]
                handles = [
                    plt.Line2D([0], [0], marker='.', color='w',
                               markerfacecolor=f"C{i}", markersize=10)
                    for i in range(N_CLASSES)
                ]
                ax1.legend(handles, legend_labels, title="Classes",
                           fontsize=18, title_fontsize=20)

            y_ps_sums = y_ps_mesh[:, :, 0].sum(axis=1)

            ax = fig.add_subplot(grid[0, i + 1])

            scatter = ax.scatter(
                X_test_mesh[:, 0],
                X_test_mesh[:, 1],
                c=y_ps_sums,
                marker='s',
                edgecolor="none",
                s=220 * step,
                alpha=1,
                cmap=color_map,
                vmin=0,
                vmax=N_CLASSES,
            )
            ax.scatter(x_test[display_ind, 0], x_test[display_ind, 1],
                       c=[f"C{x}" for x in y_test[display_ind]],
                       alpha=0.6, marker=".", edgecolor="gray", s=50)

            colorbar = plt.colorbar(scatter, ax=ax)
            colorbar.ax.set_ylabel("Set size", fontsize=20)
            colorbar.ax.tick_params(labelsize=18)
            ax.set_title(name, fontsize=22, pad=12)
            ax.set_xlim([-6, 8])
            ax.set_ylim([-6, 8])

            if isinstance(mapie, SplitCPClassifier):
                centers = []
                for f in mapie.calibrator_.functions_ + [mapie.calibrator_]:
                    if hasattr(f, "points_"):
                        centers += list(f.points_)
                if len(centers) > 0:
                    centers = np.stack(centers)
                else:
                    centers = None

                if centers is not None:
                    ax.scatter(centers[:, 0], centers[:, 1], c="gold",
                               alpha=1, edgecolors="black", s=50)

        scores[i, 1:] = [
            y_ps_test[(y_test == c), c, 0].astype(int).sum(axis=0)
            / len(y_ps_test[(y_test == c), :, 0])
            for c in range(N_CLASSES)
        ]
        scores[i, 0] = np.mean(scores[i, 1:])

    if plot:
        fig.tight_layout()
        plt.show()
    else:
        return scores


def plot_cond_coverage(scores, names):
    labels = [f"Class {i}" for i in range(N_CLASSES)]
    labels.insert(0, "marginal")
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(mapies)):
        ax.boxplot(
            scores[:, i, :], positions=x + width * (i-1), widths=width,
            patch_artist=True, boxprops=dict(facecolor=f"C{i}"),
            medianprops=dict(color="black"), labels=labels
        )
    ax.axhline(y=1-ALPHA, color='red', linestyle='--', label=f'alpha={ALPHA}')
    ax.axvline(x=0.5, color='black', linestyle='--')

    ax.set_ylabel('Coverage')
    ax.set_title('Coverage on each class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.6, 1])

    custom_handles = [Patch(facecolor=f"C{i}", edgecolor='black',
                            label=names[i]) for i in range(len(mapies))]
    handles, labels = ax.get_legend_handles_labels()

    # Update the legend with the combined handles and labels
    ax.legend(handles + custom_handles, labels + names, loc="lower left")

    plt.show()


##############################################################################
# 3. Creation of Mapie instances
# --------------------------------------------------------------------------
# We are going to compare the standard ``LAC`` method with:
#
# - The ``CCP`` method using the predicted classes as groups (to have a
#   homogenous coverage on each class).
# - The ``CCP`` method with gaussian kernels, to have adaptative prediction
#   sets, without prior knowledge or information
#   (:class:`~mapie.future.calibrators.ccp.GaussianCCP`).


n_train = 5000
n_calib = 3000
n_test = 10000

cv = ShuffleSplit(n_splits=1, test_size=n_calib/(n_train + n_calib),
                  random_state=random_state)

# =========================== Standard LAC ===========================
mapie_lac = MapieClassifier(LogisticRegression(), method="lac", cv=cv)


# ============= CCP indicator groups on predicted classes =============
mapie_ccp_y_pred = SplitCPClassifier(
    LogisticRegression(),
    calibrator=CustomCCP(lambda y_pred: y_pred),
    alpha=ALPHA, cv=cv, conformity_score=LACConformityScore()
)

# ======================== CCP Gaussian kernels ========================
mapie_ccp_gauss = SplitCPClassifier(
    LogisticRegression(),
    calibrator=GaussianCCP(40, 1, bias=True),
    alpha=ALPHA, cv=cv, conformity_score=LACConformityScore()
)

mapies = [mapie_lac, mapie_ccp_y_pred, mapie_ccp_gauss]
names = ["Standard LAC", "CCP predicted class groups", "CCP Gaussian kernel"]


##############################################################################
# 4. Generate the prediction sets
# --------------------------------------------------------------------------

run_exp(mapies, names, ALPHA, n_train=n_train, n_calib=n_calib, n_test=n_test)

##############################################################################
# We can see that the ``CCP`` method seems to create better
# prediction sets than the standard method. Indeed, where the
# classes distributions overlap (especially for class 3 and 4),
# the size of the sets should increase, to correctly represente the model
# uncertainty on those samples.
#
# The middle of all the classes distributions, where points could
# belong to any class, should have the biggest prediction sets (with almost
# all the clases in the sets, as we are very uncertain). The calibrator
# with gaussian kernels perfectly represented this uncertainty, with big sets
# for the middle points (the dark purple being sets with 4 classes).
#
# Thus, between the two ``CCP`` methods, the one using gaussian kernels
# (:class:`~mapie.future.calibrators.ccp.GaussianCCP`) seems the most
# adaptative.
#
# This modelisation of uncertainty is not visible at all in the standard
# method, where we have, in the opposite, empty sets where the distributions
# overlap.


##############################################################################
# 5. Evaluate the adaptativity
# --------------------------------------------------------------------------
# If we can, at first, assess the adaptativity of the methods just looking at
# the prediction sets, the most accurate way is to look if the coverage is
# homogenous on sub parts of the data (on each class for instance).


N_TRIALS = 6
scores = np.zeros((N_TRIALS, len(mapies), N_CLASSES+1))
for i in range(N_TRIALS):
    scores[i, :, :] = run_exp(
        mapies, names, ALPHA, n_train=n_train, n_calib=n_calib, n_test=n_test,
        plot=False, seed=i
    )

plot_cond_coverage(scores, names)

##############################################################################
# A pefectly adaptative method whould result in a homogenous coverage
# for all classes. We can see that the ``CCP`` method, with the predicted
# classes as groups, is more adaptative than the standard method. The
# over-coverage of the standard method on class 1 was corrected in the ``CCP``
# method, and the under-coverage on class 4 was also slightly corrected.
#
# However, the ``CCP`` with a gaussian calibrator
# (:class:`~mapie.future.calibrators.ccp.GaussianCCP`), is clearly the
# most adaptative method, with no under-coverage neither for the class 2 and 4.
#
# To conclude, the ``CCP`` method offer adaptative perdiction sets.
# We can inject prior knowledge or groups on which we want to avois bias
# (We tried to do this with the classes, but it was not perfect because we only
# had access to the predictions, not the true classes).
# Using gaussian kernels, with a correct sigma parameter
# (which can be optimized using cross-validation if needed), can be the easiest
# and best solution to have very adaptative prdiction sets.
