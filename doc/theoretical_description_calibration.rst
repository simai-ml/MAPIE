.. title:: Theoretical Description : contents

.. _theoretical_description_calibration:

#######################
Theoretical Description
#######################

Note: in theoretical parts of the documentation, we use the following terms employed in the scientific literature:

- :math:`alpha` is equivalent to :math:`1-confidence_level`. It can be seen as a *risk level*
- *calibrate* and *calibration*, are equivalent to *conformalize* and *conformalization*.

—

One method for multi-class calibration has been implemented in MAPIE so far :
Top-Label Calibration [1].

The goal of binary calibration is to transform a score (typically given by an ML model) that is not a probability into a
probability. The algorithms that are used for calibration can be interpreted as estimators of the confidence level. Hence,
they need independent and dependent variables to be fitted.

The figure below illustrates what we would expect as a result of a calibration, with the scores predicted being closer to the
true probability compared to the original output.

.. image:: images/calibration_basic.png
   :width: 300
   :align: center


Firstly, we introduce binary calibration, we denote the :math:`(h(X), y)` pair as the score and ground truth for the object. Hence, :math:`y`
values are in :math:`{0, 1}`. The model is calibrated if for every output :math:`q \in [0, 1]`, we have:

.. math:: 
    Pr(Y = 1 \mid h(X) = q) = q

where :math:`h()` is the score predictor.

To apply calibration directly to a multi-class context, Gupta et al. propose a framework, multiclass-to-binary, in order to reduce
a multi-class calibration to multiple binary calibrations (M2B).


Top-Label
---------

Top-Label calibration is a calibration technique introduced by Gupta et al. to calibrate the model according to the highest score and
the corresponding class (see [1] Section 2). This framework offers to apply binary calibration techniques to multi-class calibration.

More intuitively, top-label calibration simply performs a binary calibration (such as Platt scaling or isotonic regression) on the
highest score and the corresponding class, whereas confidence calibration only calibrates on the highest score (see [1] Section 2).

Let :math:`c` be the classifier and :math:`h` be the maximum score from the classifier. The couple :math:`(c, h)` is calibrated
according to Top-Label calibration if:

.. math:: 
    Pr(Y = c(X) \mid h(X), c(X)) = h(X)


References
----------

[1] Gupta, Chirag, and Aaditya K. Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
arXiv preprint arXiv:2107.08353 (2021).

[2] Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
Metrics of calibration for probabilistic predictions.
The Journal of Machine Learning Research.
2022 Jan 1;23(1):15886-940.

[3] Tygert M.
Calibration of P-values for calibration and for deviation
of a subpopulation from the full population.
arXiv preprint arXiv:2202.00100.
2022 Jan 31.

[4] D. A. Darling. A. J. F. Siegert.
The First Passage Problem for a Continuous Markov Process.
Ann. Math. Statist. 24 (4) 624 - 639, December, 1953.

[5] William Feller.
The Asymptotic Distribution of the Range of Sums of
Independent Random Variables.
Ann. Math. Statist. 22 (3) 427 - 432
September, 1951.

[6] Spiegelhalter DJ.
Probabilistic prediction in patient management and clinical trials.
Statistics in medicine.
1986 Sep;5(5):421-33.