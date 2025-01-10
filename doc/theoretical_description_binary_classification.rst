.. title:: Theoretical Description Binary Classification : contents

.. _theoretical_description_binay_classification:

#######################
Theoretical Description
#######################

Note: in theoretical parts of the documentation, we use the following terms employed in the scientific literature:

- :math:`alpha` is equivalent to :math:`1-confidence_level`. It can be seen as a *risk level*
- *calibrate* and *calibration*, are equivalent to *conformalize* and *conformalization*.

—

There are mainly three different ways to handle uncertainty quantification in binary classification:
calibration (see :doc:`theoretical_description_calibration`), confidence interval (CI) for the probability
:math:`P(Y \vert \hat{\mu}(X))` and prediction sets (see :doc:`theoretical_description_classification`).
These 3 notions are tightly related for score-based classifier, as it is shown in [1]. 

Prediction sets can be computed in the same way for multiclass and binary classification with
:class:`~mapie.classification.MapieClassifier`, and there are the same theoretical guarantees.
Nevertheless, prediction sets are often much less informative in the binary case than in the multiclass case.

From Gupta et al [1]:

    PSs and CIs are only ‘informative’ if the sets or intervals produced by them are small. To quantify
    this, we measure CIs using their width (denoted as :math:`|C(.)|)`, and PSs using their diameter (defined as
    the width of the convex hull of the PS). For example, in the case of binary classification, the diameter
    of a PS is :math:`1` if the prediction set is :math:`\{0,1\}`, and :math:`0` otherwise (since :math:`Y\in\{0,1\}`
    always holds, the set :math:`\{0,1\}` is ‘uninformative’). A short CI such as :math:`[0.39, 0.41]`
    is more informative than a wider one such as :math:`[0.3, 0.5]`.

In a few words, what you need to remember about these concepts :

* *Calibration* is useful for transforming a score (typically given by an ML model)
  into the probability of making a good prediction.
* *Set Prediction* gives the set of likely predictions with a probabilistic guarantee that the true label is in this set.
* *Probabilistic Prediction* gives a confidence interval for the predictive distribution.


1. Set Prediction
-----------------

Definition 1 (Prediction Set (PS) w.r.t :math:`f`) [1].
    Fix a predictor :math:`\hat{\mu}:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y}) \sim P`.
    Define the set of all subsets of :math:`\mathcal{Y}`, :math:`L = \{\{0\}, \{1\}, \{0, 1\}, \emptyset\}`.
    A function :math:`S:[0,1]\to\mathcal{L}` is said to be :math:`(1-\alpha)`-PS with respect to :math:`\hat{\mu}` if:

.. math:: 
    P(Y\in S(\hat{\mu}(X))) \geq 1 - \alpha

PSs are typically studied for larger output sets, such as :math:`\mathcal{Y}_{regression}=\mathbb{R}` or
:math:`\mathcal{Y}_{multiclass}=\{1, 2, ..., L > 2\}`.

See :class:`~mapie.classification.MapieClassifier` to use a set predictor.


2. Probabilistic Prediction
---------------------------

Definition 2 (Confidence Interval (CI) w.r.t :math:`\hat{\mu}`) [1].
    Fix a predictor :math:`\hat{\mu}:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y}) \sim P`.
    Let :math:`I` denote the set of all subintervals of :math:`[0,1]`.
    A function :math:`C:[0,1]\to\mathcal{I}` is said to be :math:`(1-\alpha)`-CI with respect to :math:`\hat{\mu}` if:

.. math:: 
    P(\mathbb{E}[Y|\hat{\mu}(X)]\in C(\hat{\mu}(X))) \geq 1 - \alpha


3. Calibration
--------------

Usually, calibration is understood as perfect calibration meaning (see :doc:`theoretical_description_calibration`).
In practice, it is more reasonable to consider approximate calibration.

Definition 3 (Approximate calibration) [1].
    Fix a predictor :math:`\hat{\mu}:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y}) \sim P`.
    The predictor :math:`\hat{\mu}:\mathcal{X} \to [0, 1]` is :math:`(\epsilon,\alpha)`-calibrated
    for some :math:`\epsilon,\alpha\in[0, 1]` if with probability at least :math:`1-\alpha`:

.. math:: 
    |\mathbb{E}[Y|\hat{\mu}(X)] - \hat{\mu}(X)| \leq \epsilon

See :class:`~sklearn.calibration.CalibratedClassifierCV` or :class:`~mapie.calibration.MapieCalibrator`
to use a calibrator.

In the CP framework, it is worth noting that Venn predictors produce probability-type predictions
for the labels of test objects which are guaranteed to be well-calibrated under the standard assumption
that the observations are generated independently from the same distribution [2].


References
----------

[1] Gupta, Chirag, Aleksandr Podkopaev, and Aaditya Ramdas.
"Distribution-free binary classification: prediction sets, confidence intervals, and calibration."
Advances in Neural Information Processing Systems 33 (2020): 3711-3723.

[2] Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer.
"Algorithmic Learning in a Random World."
Springer Nature, 2022.