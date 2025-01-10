.. title:: Theoretical Description Multi label Classification : contents

.. _theoretical_description_multilabel_classification:

#######################
Theoretical Description
#######################

Note: in theoretical parts of the documentation, we use the following terms employed in the scientific literature:

- :math:`alpha` is equivalent to :math:`1-confidence_level`. It can be seen as a *risk level*
- *calibrate* and *calibration*, are equivalent to *conformalize* and *conformalization*.

—

Three methods for multi-label uncertainty quantification have been implemented in MAPIE so far :
Risk-Controlling Prediction Sets (RCPS) [1], Conformal Risk Control (CRC) [2] and Learn Then Test (LTT) [3].
The difference between these methods is the way the conformity scores are computed. 

For a multi-label classification problem in a standard independent and identically distributed (i.i.d) case,
our training data :math:`(X, Y) = \{(x_1, y_1), \ldots, (x_n, y_n)\}`` has an unknown distribution :math:`P_{X, Y}`. 

For any risk level :math:`\alpha` between 0 and 1, the methods implemented in MAPIE allow the user to construct a prediction
set :math:`\hat{C}_{n, \alpha}(X_{n+1})` for a new observation :math:`\left( X_{n+1},Y_{n+1} \right)` with a guarantee
on the recall. RCPS, LTT, and CRC give three slightly different guarantees:

- RCPS:

.. math::
    \mathbb{P}(R(\mathcal{T}_{\hat{\lambda}}) \leq \alpha ) \geq 1 - \delta

- CRC:

.. math::
    \mathbb{E}\left[L_{n+1}(\hat{\lambda})\right] \leq \alpha

- LTT:

.. math::
    \mathbb{P}(R(\mathcal{T}_{\hat{\lambda}}) \leq \alpha ) \geq 1 - \delta \quad \texttt{with} \quad p_{\hat{\lambda}} \leq \frac{\delta}{\lvert \Lambda \rvert}


Notice that at the opposite of the other two methods, LTT allows to control any non-monotone loss. In MAPIE for multilabel classification,
we use CRC and RCPS for recall control and LTT for precision control.

1. Risk-Controlling Prediction Sets
===================================
1.1. General settings
---------------------


Let's first give the settings and the notations of the method:

- Let :math:`\mathcal{T}_{\hat{\lambda}}: X \longrightarrow Y'` be a set-valued function (a tolerance region) that maps a feature vector to a set-valued prediction. This function is built from the model which was previously fitted on the training data. It is indexed by a one-dimensional parameter :math:`\lambda` which is taking values in :math:`\Lambda \subset \mathbb{R} \cup \{ \pm \infty \}` such that:

.. math::
   \lambda_1 < \lambda_2 \Rightarrow \mathcal{T}_{\lambda_1}(x) \subset \mathcal{T}_{\lambda_2}(x)

- Let :math:`L: Y\times Y' \longrightarrow \mathbb{R}^+` be a loss function on a prediction set with the following nesting property:

.. math::
   S_1 \subset S_2 \Rightarrow L(y, S_1) \geq L(y, S_2)

- Let :math:`R` be the risk associated to a set-valued predictor:

.. math::
    R(\mathcal{T}_{\hat{\lambda}}) = \mathbb{E}[L(Y, \mathcal{T}_{\lambda}(X))]

The goal of the method is to compute an Upper Confidence Bound (UCB) :math:`\hat{R}^+(\lambda)` of :math:`R(\lambda)` and then to find
:math:`\hat{\lambda}` as follows:

.. math::
    \hat{\lambda} = \inf\{\lambda \in \Lambda: \hat{R}^+(\lambda ') < \alpha, \forall \lambda ' \geq \lambda \}

The figure below explains this procedure:

.. image:: images/r_hat_plus.png
   :width: 600
   :align: center

Following those settings, the RCPS method gives the following guarantee on the recall:

.. math::
    \mathbb{P}(R(\mathcal{T}_{\hat{\lambda}}) \leq \alpha ) \geq 1 - \delta


1.2. Bounds calculation
-----------------------

In this section, we will consider only bounded losses (as for now, only the :math:`1-recall` loss is implemented).
We will show three different Upper Calibration Bounds (UCB) (Hoeffding, Bernstein, and Waudby-Smith–Ramdas) of :math:`R(\lambda)`
based on the empirical risk which is defined as follows:

.. math::
    \hat{R}(\lambda) = \frac{1}{n}\sum_{i=1}^n L(Y_i, T_{\lambda}(X_i))


1.2.1. Hoeffding Bound
----------------------

Suppose the loss is bounded above by one, then we have by the Hoeffding inequality that:

.. math::
    P((\hat{R}(\lambda)-R(\lambda) \leq -x)) = \exp\{-2nx^2\}

Which implies the following UCB:

.. math::
    \hat{R}_{Hoeffding}^+(\lambda) = \hat{R}(\lambda) + \sqrt{\frac{1}{2n}\log\frac{1}{\delta}}


1.2.2. Bernstein Bound
----------------------

Contrary to the Hoeffding bound, which can sometimes be too simple, the Bernstein UCB takes into account the variance
and gives a smaller prediction set size:

.. math::
    \hat{R}_{Bernstein}^+(\lambda) = \hat{R}(\lambda) + \hat{\sigma}(\lambda)\sqrt{\frac{2\log(2/\delta)}{n}} + \frac{7\log (2/\delta)}{3(n-1)}

Where:

.. math::
    \hat{\sigma}(\lambda) = \frac{1}{n-1}\sum_{i=1}^n(L(Y_i, T_{\lambda}(X_i)) - \hat{R}(\lambda))^2


1.2.3. Waudby-Smith–Ramdas
--------------------------

This last UCB is the one recommended by the authors of [1] to use when using a bounded loss as this is the one that gives
the smallest prediction sets size while having the same risk guarantees. This UCB is defined as follows:

Let :math:`L_i (\lambda) = L(Y_i, T_{\lambda}(X_i))` and

.. math::
    \hat{\mu}_i (\lambda) = \frac{1/2 + \sum_{j=1}^i L_j (\lambda)}{1 + i},
    \hat{\sigma}_i^2 (\lambda) = \frac{1/4 + \sum_{j=1}^i (L_j (\lambda) - \hat{\mu}_i (\lambda))}{1 + i},
    \nu_i (\lambda) = \min \left\{ 1, \sqrt{\frac{2\log (1/\delta)}{n \hat{\sigma}_{i-1}^2 (\lambda)}}\right\}

Further let:

.. math::
    K_i(R, \lambda) = \prod_{j=1}^i\{1 - \nu_j(\lambda)(L_j (\lambda) - R)\}

Then:

.. math::
    \hat{R}_{WSR}^+(\lambda) = \inf \{ R \geq 0 : \max_{i=1,...n} K_i(R, \lambda) > \frac{1}{\delta}\}


2. Conformal Risk Control
=========================

The goal of this method is to control any monotone and bounded loss. The result of this method can be expressed as follows:

.. math::
    \mathbb{E}\left[L_{n+1}(\hat{\lambda})\right] \leq \alpha

Where :math:`L_{i}(\lambda) = l(C_{\lambda}(X_{i}), Y_{i})`

In the case of multi-label classification, :math:`C_{\lambda}(x) = \{ k : f(X)_k \geq 1 - \lambda \}`

To find the optimal value of :math:`\lambda`, the following algorithm is applied:

.. math::
    \hat{\lambda} = \inf \{ \lambda: \frac{n}{n + 1}\hat{R}_n (\lambda) + \frac{B}{n + 1} \leq \alpha \}

With :

.. math::
    \hat{R}_n (\lambda) = (L_{1}(\lambda) + ... + L_{n}(\lambda)) / n


3. Learn Then Test
==================

We are going to present the Learn Then Test framework that allows the user to control non-monotonic risk such as precision score.
This method has been introduced in article [3].
The settings here are the same as RCPS and CRC, we just need to introduce some new parameters:

- Let :math:`\Lambda` be a discretized for our :math:`\lambda`, meaning that :math:`\Lambda = \{\lambda_1, ..., \lambda_n\}`.

- Let :math:`p_\lambda` be a valid p-value for the null hypothesis :math:`\mathbb{H}_j: R(\lambda_j)>\alpha`.

The goal of this method is to control any loss whether monotonic, bounded, or not, by performing risk control through multiple
hypothesis testing. We can express the goal of the procedure as follows:

.. math::
    \mathbb{P}(R(\mathcal{T}_{\lambda}) \leq \alpha ) \geq 1 - \delta

In order to find all the parameters :math:`\lambda` that satisfy the above condition, the Learn Then Test framework proposes to do the following:

- First across the collections of functions :math:`(T_\lambda)_{\lambda\in\Lambda}`, we estimate the risk on the calibration data
  :math:`\{(x_1, y_1), \dots, (x_n, y_n)\}`.

- For each :math:`\lambda_j` in a discrete set :math:`\Lambda = \{\lambda_1, \lambda_2,\dots, \lambda_n\}`, we associate the null hypothesis
  :math:`\mathcal{H}_j: R(\lambda_j) > \alpha`, as rejecting the hypothesis corresponds to selecting :math:`\lambda_j` as a point where risk the risk 
  is controlled.

- For each null hypothesis, we compute a valid p-value using a concentration inequality :math:`p_{\lambda_j}`. Here we choose to compute the Hoeffding-Bentkus p-value
  introduced in the paper [3].

- Return :math:`\hat{\Lambda} =  \mathcal{A}(\{p_j\}_{j\in\{1,\dots,\lvert \Lambda \rvert})`, where :math:`\mathcal{A}`, is an algorithm
  that controls the family-wise error rate (FWER), for example, Bonferonni correction.


References
==========

[1] Lihua Lei Jitendra Malik Stephen Bates, Anastasios Angelopoulos,
and Michael I. Jordan. Distribution-free, risk-controlling prediction
sets. CoRR, abs/2101.02703, 2021. URL https://arxiv.org/abs/2101.02703.39

[2] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua,
Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

[3] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
M. I., & Lei, L. (2021). Learn then test:
"Calibrating predictive algorithms to achieve risk control".
