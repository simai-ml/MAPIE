.. title:: Theoretical Description : contents

.. _theoretical_description_ccp:

########################
Theoretical Description
########################

The Conditional Conformal Prediction (CCP) method :ref:`[1]<theoretical_description_ccp_references>` is a model agnostic conformal prediction method which
can create adaptative prediction intervals.

In MAPIE, this method has a lot of advantages:

- It is model agnostic (it doesn’t depend on the model but only on the predictions, unlike CQR)
- It can create very adaptative intervals (with a varying width which truly reflects the model uncertainty)
- while providing coverage guarantee on all sub-groups of interest (avoiding biases)
- with the possibility to inject prior knowledge about the data or the model

However, we will also see its disadvantages:
- The adaptativity depends on the calibrator we use: It can be difficult to choose the correct calibrator,
with the best parameters.
- The calibration and even more the inference are much longer than for the other methods.
We can reduce the inference time using ``unsafe_approximation=True``,
but we lose the strong theoretical guarantees and risk a small miscoverage
(even if, most of the time, the coverage is achieved).

To conclude, it can create more adaptative intervals than the other methods,
but it can be difficult to find the best settings (calibrator type and parameters)
and can have a big computational time.

How does it works?
====================

Method's intuition
--------------------

We recall that the `standard split method` estimates the absolute residuals by a constant :math:`\hat{q}_{n, \alpha}^+`
(which is the quantile of :math:`{|Y_i-\hat{\mu}(X_i)|}_{1 \leq i \leq n}`). Then, the prediction interval is:

.. math:: \hat{C}_{n, \alpha}^{\textrm split}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}_{n, \alpha}^+

The idea of the `CCP` method, is to learn, not a constant, but a function :math:`q(X)`,
to have a different interval width depending on the :math:`X` value. Then, we would have:

.. math:: \hat{C}_{n, \alpha}^{\textrm CCP}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}(X_{n+1})

To be able to find the best function, while having some coverage guarantees,
we should select this function inside some defined class of functions :math:`\mathcal{F}`.

This method is motivated by the following equivalence:

.. math:: 
  \begin{array}{c}
  \mathbb{P}(Y_{n+1} \in \hat{C} \; | \; X_{n+1}=x) = 1 - \alpha, \quad \text{for all x} \\
  \textstyle \Longleftrightarrow \\
  \mathbb{E} \left[ f(X_{n+1}) \mathbb{I} \left\{ Y_{n+1} \in \hat{C}(X_{n+1}) \right\} \right] = 0, \quad \text{for all measurable f} \\
  \end{array}

This is the equation corresponding to the perfect conditional coverage, which is theoretically impossible to obtain.
Then, relaxing this objective by replacing "all measurable f" with "all f belonging to some class :math:`\mathcal{F}`"
seems a way to get close to the perfect conditional coverage.


.. _theoretical_description_ccp_control_steps:

The method follow 3 steps:
----------------------------

1. Choose  a class of functions. The simple approach is to choose a class a finite dimension :math:`d \in \mathbb{N}`,
   using, for any :math:`\Phi \; : \; \mathbb{R}^d \to \mathbb{R}` (chosen by the user)

  .. math::
    \mathcal{F} = \left\{ \Phi (\cdot)^T \beta  :  \beta \in \mathbb{R}^d \right\}

2. Find the best function of this class by solving the following optimization problem:

  .. note:: It is actually a quantile regression between the transformation :math:`\Phi (X)` and the conformity scores `S`.
  
  Considering an upper bound :math:`M` of the conformity scores,
  such as :math:`S_{n+1} < M`:

  .. math::
    \hat{g}_M^{n+1} := \text{arg}\min_{g \in \mathcal{F}} \; \frac{1}{n+1} \sum_{i=1}^n{l_{\alpha} (g(X_i), S_i)} \; + \frac{1}{n+1}l_{\alpha} (g(X_{n+1}), M)

  .. warning::
    In the :ref:`API<api>`, we use by default :math:`M=max(\{S_i\}_{i\leq n})`,
    the maximum conformity score of the calibration set,
    but you can specify it yourself if a bound is known, considering your data,
    model and conformity score.

    Moreover, it means that there is still small computations which are done
    for each test point :math:`X_{n+1}`. If you want to avoid that, you can
    use ``unsafe_approximation=True``, which only consider:
    
    .. math::
      \hat{g} :=  \text{arg}\min_{g \in \mathcal{F}} \; \frac{1}{n} \sum_{i=1}^n{l_{\alpha} (g(X_i), S_i)}

    However, it may result in a small miscoverage.
    It is recommanded to empirically check the resulting coverage on the test set.

3. We use this optimized function :math:`\hat{g}_M^{n+1}` to compute the prediction intervals:
  
  .. math::
    \hat{C}_M^{n+1}(X_{n+1}) = \{ y : S(X_{n+1}, \: y) \leq \hat{g}_M^{n+1}(X_{n+1}) \}

  .. note:: The formulas are generic and work with all conformity scores. But in the case of the absolute residuals, we get:
    
    .. math::
      \hat{C}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{g}_M^{n+1}(X_{n+1})

.. _theoretical_description_ccp_control_coverage:

Coverage guarantees:
-----------------------

.. warning::
  The following guarantees assume that the approximation described above is not used, and that
  the chosen bound M is indeed such as :math:`\forall \text{ test index }i, \; S_i < M`

Following this steps, we have the coverage guarantee:
:math:`\forall f \in \mathcal{F},`

.. math::
  \mathbb{P}_f(Y_{n+1} \in \hat{C}_M^{n+1}(X_{n+1})) \geq 1 - \alpha

.. math::
  \text{and} \quad \left | \mathbb{E} \left[ f(X_{n+1}) \left(\mathbb{I} \left\{ Y_{n+1} \in \hat{C}_M^{n+1}(X_{n+1}) \right\} - (1 - \alpha) \right) \right] \right |
  \leq \frac{d}{n+1} \mathbb{E} \left[ \max_{1 \leq i \leq n+1} \left|f(X_i)\right| \right]

.. note:: 
  If we want to have a homogenous coverage on some given groups in :math:`\mathcal{G}`, we can use
  :math:`\mathcal{F} = \{ x \mapsto \sum _{G \in \mathcal{G}} \; \beta_G \mathbb{I} \{ x \in G \} : \beta_G \in \mathbb{R} \}`,
  then we have :math:`\forall G \in \mathcal{G}`:

  .. math::
    1 - \alpha
    \leq \mathbb{P} \left( Y_{n+1} \in \hat{C}_M^{n+1}(X_{n+1}) \; | \; X_{n+1} \in G \right) 
    \leq 1- \alpha + \frac{|\mathcal{G}|}{(n+1) \mathbb{P}(X_{n+1} \in G)} \\
    = 1- \alpha + \frac{\text{number of groups in } \mathcal{G}}{\text{number of samples of } \{X_i\} \text{ in G}}

How to use it in practice?
============================

Creating a class a function adapted to our needs
--------------------------------------------------

The following will provide some tips on how to use the method (for more practical examples, see
:doc:`examples_regression/4-tutorials/plot_ccp_tutorial` or
`How to leverage the CCP method on real data
<https://github.com/scikit-learn-contrib/MAPIE/tree/master/notebooks/regression/tutorial_ccp_CandC.ipynb>`_
).

1. If you want a generally adaptative interval and you don't have prior
   knowledge about your data, you can use gaussian kernels, implemented in Mapie
   in :class:`~mapie.future.calibrators.ccp.GaussianCCP`. See the API doc for more information.

2. If you want to avoid bias on sub-groups and ensure an homogenous coverage on those,
   you can add indicator functions corresponding to those groups. 

3. You can inject prior knowledge in the method using :class:`~mapie.future.calibrators.ccp.CustomCCP`,
   if you have information about the conformity scores distribution
   (domains with different biavior, expected model uncertainty depending on a given feature, etc).

4. Empirically test obtained coverage on a test set, to make sure that the expected coverage is achieved. 


Avoid miscoverage
--------------------

- | To guarantee marginal coverage, you need to have an intercept term in the :math:`\Phi` function (meaning, a feature equal to :math:`1` for all :math:`X_i`).
  | It correspond, in the :ref:`API<api>`, to ``bias=True``.

- | Some miscoverage can come from the optimization process, which is
    solved with numerical methods, and may fail to find the global minimum.
    If the target coverage is not achieved, you can try adding regularization,
    to help the optimization process. You can also try reducing the number of dimensions :math:`d`
    or using a smoother :math:`\Phi` function, such as with gaussian kernels
    (indeed, using only indicator functions makes the optimization difficult).

    .. warning::
      Adding some regularization will theoretically induce a miscoverage,
      as the objective function will slightly increase, to minimize the regularization term.
      
      In practice, it may increase the coverage (as it helps the optimization convergence),
      but it can also decrease it. Always empirically check the resulting coverage
      and avoid too big regularization terms (below :math:`10^{-4}` is usually recommanded).


- | Finally, if you have coverage issues because the optimisation is difficult,
    you can artificially enforce higher coverage by reducing the value of :math:`\alpha`.
    Evaluating the best adjusted :math:`\alpha` using cross-validation will ensure
    the same coverage on the test set (subject to variability due to the finite number of samples).


.. _theoretical_description_ccp_references:

References
==========

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees", `arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.