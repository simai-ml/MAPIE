.. title:: How to measure conformal prediction performance?

.. _theoretical_description_metrics:

#######################
Theoretical Description
#######################

Note: in theoretical parts of the documentation, we use the following terms employed in the scientific literature:

- :math:`alpha` is equivalent to :math:`1-confidence_level`. It can be seen as a *risk level*
- *calibrate* and *calibration*, are equivalent to *conformalize* and *conformalization*.

—

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.

1. General Metrics
==================

Regression Coverage Score
-------------------------

The **Regression Coverage Score (RCS)** calculates the fraction of true outcomes that fall within the provided prediction intervals.

.. math::

   RCS = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(\hat y^{\text{low}}_{i} \leq y_{i} \leq \hat y^{\text{up}}_{i})

where:

- :math:`n` is the number of samples,
- :math:`y_{i}` is the true value for the :math:`i`-th sample,
- :math:`\hat y^{\text{low}}_{i}` and :math:`\hat y^{\text{up}}_{i}` are the lower and upper bounds of the prediction intervals, respectively.

Regression Mean Width Score
---------------------------

The **Regression Mean Width Score (RMWS)** assesses the average width of the prediction intervals provided by the model.

.. math::

   \text{RMWS} = \frac{1}{n} \sum_{i=1}^{n} (\hat y^{\text{up}}_{i} - \hat y^{\text{low}}_{i})

Classification Coverage Score
-----------------------------

The **Classification Coverage Score (CCS)** measures how often the true class labels fall within the predicted sets.

.. math::

   CCS = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(y_{i} \in \hat C(x_{i}))

Here, :math:`\hat C(x_{i})` represents the set of predicted labels that could possibly contain the true label for the :math:`i`-th observation :math:`x_{i}`.

Classification Mean Width Score
-------------------------------

For classification tasks, the **Classification Mean Width Score (CMWS)** calculates the average size of the prediction sets across all samples.

.. math::

   \text{CMWS} = \frac{1}{n} \sum_{i=1}^{n} |\hat C(x_i)|

where :math:`|\hat C(x_i)|` denotes the number of classes included in the prediction set for sample :math:`i`.

Size-Stratified Coverage
-------------------------

**Size-Stratified Coverage (SSC)** evaluates how the size of prediction sets or intervals affects their ability to cover the true outcomes [1]. It's calculated separately for classification and regression:

**Regression:**

.. math::

   \text{SSC}_{\text{regression}} = \sum_{k=1}^{K} \left( \frac{1}{|I_k|} \sum_{i \in I_k} \mathbf{1}(\hat y^{\text{low}}_{i} \leq y_{i} \leq \hat y^{\text{up}}_{i}) \right)

**Classification:**

.. math::

   \text{SSC}_{\text{classification}} = \sum_{k=1}^{K} \left( \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{1}(y_{i} \in \hat C(x_i)) \right)

where:

- :math:`K` is the number of distinct size groups,
- :math:`I_k` and :math:`S_k` are the indices of samples whose prediction intervals or sets belong to the :math:`k`-th size group.

Hilbert-Schmidt Independence Criterion
---------------------------------------

The **Hilbert-Schmidt Independence Criterion (HSIC)** is a non-parametric measure of independence between two variables, applied here to test the independence of interval sizes from their coverage indicators [4].

.. math::

   \text{HSIC} = \operatorname{trace}(\mathbf{H} \mathbf{K} \mathbf{H} \mathbf{L})

where:

- :math:`\mathbf{K}` and :math:`\mathbf{L}` are the kernel matrices representing the interval sizes and coverage indicators, respectively.
- :math:`\mathbf{H}` is the centering matrix, :math:`\mathbf{H} = \mathbf{I} - \frac{1}{n} \mathbf{11}^\top`.

This measure is crucial for determining whether certain sizes of prediction intervals are systematically more or less likely to contain the true values, which can highlight biases in interval-based predictions.

Coverage Width-Based Criterion
------------------------------

The **Coverage Width-Based Criterion (CWC)** evaluates prediction intervals by balancing their empirical coverage and width. It is designed to both reward narrow intervals and penalize those that do not achieve a specified coverage probability [6].

.. math::

   \text{CWC} = (1 - \text{Mean Width Score}) \times \exp\left(-\eta \times (\text{Coverage Score} - (1-\alpha))^2\right)

Mean Winkler Interval Score
---------------------------

The **Mean Winkler Interval (MWI) Score** evaluates prediction intervals by combining their width with a penalty for intervals that do not contain the observation [8, 10].

.. math::

   \text{MWI Score} = \frac{1}{n} \sum_{i=1}^{n} (\hat y^{\text{up}}_{i} - \hat y^{\text{low}}_{i}) + \frac{2}{\alpha} \sum_{i=1}^{n} \max(0, |y_{i} - \hat y^{\text{boundary}}_{i}|)

where :math:`\hat y^{\text{boundary}}_{i}` is the nearest interval boundary not containing :math:`y_{i}`, and :math:`\alpha` is the significance level.

2. Calibration Metrics
======================


Expected Calibration Error
--------------------------

The **Expected Calibration Error** (ECE) is a metric used to evaluate how well the predicted probabilities of a model align with the actual outcomes. It measures the difference between predicted confidence levels and actual accuracy. The process involves dividing the predictions into bins based on confidence scores and then comparing the accuracy within each bin to the average confidence level of the predictions in that bin. The number of bins is a hyperparameter :math:`M`, and we refer to a specific bin by :math:`B_m`.

For each bin :math:`B_m`, the accuracy and confidence are defined as follows:

.. math::

    \text{acc}(B_m) = \frac{1}{\left| B_m \right|} \sum_{i \in B_m} y_i

.. math::

    \text{conf}(B_m) = \frac{1}{\left| B_m \right|} \sum_{i \in B_m} \hat{f}(x_i)

The ECE is then calculated using the following formula:

.. math::

    \text{ECE} = \sum_{m=1}^M \frac{\left| B_m \right|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|

where:

- :math:`B_m` is the set of indices of samples that fall into the :math:`m`-th bin.
- :math:`\left| B_m \right|` is the number of samples in the :math:`m`-th bin.
- :math:`n` is the total number of samples.
- :math:`\text{acc}(B_m)` is the accuracy within the :math:`m`-th bin.
- :math:`\text{conf}(B_m)` is the average confidence score within the :math:`m`-th bin.

In simple terms, once the different bins from the confidence scores have been created, we check the mean accuracy of each bin. The absolute mean difference between the two is the ECE. Hence, the lower the ECE, the better the calibration was performed. The difference between the average confidence and the actual accuracy within each bin is weighted by the proportion of samples in that bin, ensuring that bins with more samples have a larger influence on the final ECE value.

Top-Label Expected Calibration Error (Top-Label ECE)
----------------------------------------------------

The **Top-Label Expected Calibration Error** (Top-Label ECE) extends the concept of ECE to the multi-class setting. Instead of evaluating calibration over all predicted probabilities, Top-Label ECE focuses on the calibration of the most confident prediction (top-label) for each sample. For the top-label class, the calculation of the accuracy and confidence is conditioned on the top label, and the average ECE is taken for each top-label.

The Top-Label ECE is calculated as follows:

.. math::

    \text{Top-Label ECE} = \frac{1}{L} \sum_{j=1}^L \sum_{i=1}^B \frac{|B_{i,j}|}{n_j} \left| \text{acc}(B_{i,j}) - \text{conf}(B_{i,j}) \right|

where:

- :math:`L` is the number of unique labels.
- :math:`B_{i,j}` is the set of indices of samples that fall into the :math:`i`-th bin for label :math:`j`.
- :math:`\left| B_{i,j} \right|` is the number of samples in the :math:`i`-th bin for label :math:`j`.
- :math:`n_j` is the total number of samples for label :math:`j`.
- :math:`\text{acc}(B_{i,j})` is the accuracy within the :math:`i`-th bin for label :math:`j`.
- :math:`\text{conf}(B_{i,j})` is the average confidence score within the :math:`i`-th bin for label :math:`j`.
- :math:`B` is the total number of bins.

For each label, the predictions are binned according to their confidence scores for that label. The calibration error is then calculated for each label separately and averaged across all labels to obtain the final Top-Label ECE value. This ensures that the calibration is measured specifically for the most confident prediction, which is often the most critical for decision-making in multi-class problems.

Cumulative Differences
----------------------

**Cumulative Differences** calculates the cumulative differences between sorted true values and prediction scores, helping to understand how well the prediction scores correspond to the actual outcomes when both are ordered by the score [2].

.. math::

   \text{Cumulative Differences} = \frac{1}{n} \sum_{i=1}^{n} (y_{\sigma_1(i)} - \hat y_{\sigma_2(i)})

where:

- :math:`\sigma_1` is the permutation which sorts all the true values.
- :math:`\sigma_2` is the permutation which sorts all the predicted values.

Kolmogorov-Smirnov Statistic for Calibration
--------------------------------------------

The **Kolmogorov-Smirnov test** was derived in [2, 3, 11]. The idea is to consider the cumulative differences between sorted scores :math:`s_i`
and their corresponding labels :math:`y_i` and to compare its properties to that of a standard Brownian motion. Let us consider the
cumulative differences on sorted scores: 

.. math::
    C_k = \frac{1}{N}\sum_{i=1}^k (y_i - s_i)

We also introduce a typical normalization scale :math:`\sigma`:

.. math::
    \sigma = \frac{1}{N}\sqrt{\sum_{i=1}^N s_i(1 - s_i)}

The Kolmogorov-Smirnov statistic is then defined as : 

.. math::
   G = \max|C_k|/\sigma

It can be shown [2] that, under the null hypothesis of well-calibrated scores, this quantity asymptotically (i.e. when N goes to infinity)
converges to the maximum absolute value of a standard Brownian motion over the unit interval :math:`[0, 1]`. [3, 11] also provide closed-form 
formulas for the cumulative distribution function (CDF) of the maximum absolute value of such a standard Brownian motion.
So we state the p-value associated to the statistical test of well calibration as:

.. math::
   p = 1 - CDF(G)

Kuiper's Test
-------------

The **Kuiper test** was derived in [2, 3, 11] and is very similar to Kolmogorov-Smirnov. This time, the statistic is defined as:

.. math::
   H = (\max_k|C_k| - \min_k|C_k|)/\sigma

It can be shown [2] that, under the null hypothesis of well-calibrated scores, this quantity asymptotically (i.e. when N goes to infinity)
converges to the range of a standard Brownian motion over the unit interval :math:`[0, 1]`. [3, 11] also provide closed-form 
formulas for the cumulative distribution function (CDF) of the range of such a standard Brownian motion.
So we state the p-value associated to the statistical test of well calibration as:

.. math::
   p = 1 - CDF(H)

Spiegelhalter’s Test
--------------------

The **Spiegelhalter test** was derived in [9]. It is based on a decomposition of the Brier score: 

.. math::
   B = \frac{1}{N}\sum_{i=1}^N(y_i - s_i)^2

where scores are denoted :math:`s_i` and their corresponding labels :math:`y_i`. This can be decomposed in two terms:

.. math::
   B = \frac{1}{N}\sum_{i=1}^N(y_i - s_i)(1 - 2s_i) + \frac{1}{N}\sum_{i=1}^N s_i(1 - s_i)

It can be shown that the first term has an expected value of zero under the null hypothesis of well calibration. So we interpret
the second term as the Brier score expected value :math:`E(B)` under the null hypothesis. As for the variance of the Brier score, it can be
computed as:

.. math::
   Var(B) = \frac{1}{N^2}\sum_{i=1}^N(1 - 2s_i)^2 s_i(1 - s_i)

So we can build a Z-score as follows: 

.. math::
   Z = \frac{B - E(B)}{\sqrt{Var(B)}} = \frac{\sum_{i=1}^N(y_i - s_i)(1 - 2s_i)}{\sqrt{\sum_{i=1}^N(1 - 2s_i)^2 s_i(1 - s_i)}}

This statistic follows a normal distribution of cumulative distribution CDF so that we state the associated p-value:

.. math::
   p = 1 - CDF(Z)


References
==========

[1] Angelopoulos, A. N., & Bates, S. (2021).
A gentle introduction to conformal prediction and
distribution-free uncertainty quantification.
arXiv preprint arXiv:2107.07511.

[2] Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
Metrics of calibration for probabilistic predictions.
The Journal of Machine Learning Research. 2022 Jan 1;23(1):15886-940.

[3] D. A. Darling. A. J. F. Siegert.
The First Passage Problem for a Continuous Markov Process.
Ann. Math. Statist. 24 (4) 624 - 639, December, 1953.

[4] Feldman, S., Bates, S., & Romano, Y. (2021).
Improving conditional coverage via orthogonal quantile regression.
Advances in Neural Information Processing Systems, 34, 2060-2071.

[5] Gupta, Chirag, and Aaditya K. Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
arXiv preprint arXiv:2107.08353 (2021).

[6] Khosravi, Abbas, Saeid Nahavandi, and Doug Creighton.
"Construction of optimal prediction intervals for load forecasting
problems."
IEEE Transactions on Power Systems 25.3 (2010): 1496-1503.

[7] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
"Obtaining well calibrated probabilities using bayesian binning."
Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

[8] Robert L. Winkler
"A Decision-Theoretic Approach to Interval Estimation",
Journal of the American Statistical Association,
volume 67, pages 187-191 (1972)
(https://doi.org/10.1080/01621459.1972.10481224)

[9] Spiegelhalter DJ.
Probabilistic prediction in patient management and clinical trials.
Statistics in medicine.
1986 Sep;5(5):421-33.

[10] Tilmann Gneiting and Adrian E Raftery
"Strictly Proper Scoring Rules, Prediction, and Estimation",
Journal of the American Statistical Association,
volume 102, pages 359-378 (2007)
(https://doi.org/10.1198/016214506000001437) (Section 6.2)

[11] Tygert M.
Calibration of P-values for calibration and for deviation
of a subpopulation from the full population.
arXiv preprint arXiv:2202.00100.2022 Jan 31.
