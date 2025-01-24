
.. _api:

#########
MAPIE API
#########

.. currentmodule:: mapie

Regression
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   regression.MapieRegressor
   regression.MapieQuantileRegressor
   regression.MapieTimeSeriesRegressor

Classification
==============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   classification.MapieClassifier

Multi-Label Classification
==========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   multi_label_classification.MapieMultiLabelClassifier

Calibration
===========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   calibration.MapieCalibrator

Metrics
=======

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   metrics.classification_coverage_score
   metrics.classification_coverage_score_v2
   metrics.classification_mean_width_score
   metrics.classification_ssc
   metrics.classification_ssc_score
   metrics.cumulative_differences
   metrics.expected_calibration_error
   metrics.hsic
   metrics.kolmogorov_smirnov_cdf
   metrics.kolmogorov_smirnov_p_value
   metrics.kolmogorov_smirnov_statistic
   metrics.kuiper_cdf
   metrics.kuiper_p_value
   metrics.kuiper_statistic
   metrics.length_scale
   metrics.regression_coverage_score
   metrics.regression_coverage_score_v2
   metrics.regression_mean_width_score
   metrics.regression_ssc
   metrics.regression_ssc_score
   metrics.spiegelhalter_p_value
   metrics.spiegelhalter_statistic
   metrics.top_label_ece

Conformity scores (regression)
==============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   conformity_scores.BaseRegressionScore
   conformity_scores.AbsoluteConformityScore
   conformity_scores.GammaConformityScore
   conformity_scores.ResidualNormalisedScore

Conformity scores (classification)
==================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   conformity_scores.BaseClassificationScore
   conformity_scores.NaiveConformityScore
   conformity_scores.LACConformityScore
   conformity_scores.APSConformityScore
   conformity_scores.RAPSConformityScore
   conformity_scores.TopKConformityScore

Resampling
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   subsample.BlockBootstrap
   subsample.Subsample

New Split CP class
===================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   future.split.base.SplitCP
   future.split.SplitCPRegressor
   future.split.SplitCPClassifier

Calibrators
===========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   future.calibrators.base.BaseCalibrator
   future.calibrators.StandardCalibrator
   future.calibrators.ccp.CCPCalibrator
   future.calibrators.ccp.CustomCCP
   future.calibrators.ccp.PolynomialCCP
   future.calibrators.ccp.GaussianCCP

Mondrian
========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mondrian.MondrianCP
