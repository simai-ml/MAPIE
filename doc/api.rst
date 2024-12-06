#########
MAPIE API
#########

Regression V1 (from mapie_v1)
=============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie_v1.regression.SplitConformalRegressor
   mapie_v1.regression.CrossConformalRegressor
   mapie_v1.regression.JackknifeAfterBootstrapRegressor
   mapie_v1.regression.ConformalizedQuantileRegressor

Regression (from mapie)
=======================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.regression.MapieRegressor
   mapie.regression.MapieQuantileRegressor
   mapie.regression.MapieTimeSeriesRegressor

Classification
==============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.classification.MapieClassifier

Multi-Label Classification
==========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.multi_label_classification.MapieMultiLabelClassifier

Calibration
===========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.calibration.MapieCalibrator

Metrics
=======

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.classification_coverage_score
   mapie.metrics.classification_coverage_score_v2
   mapie.metrics.classification_mean_width_score
   mapie.metrics.classification_ssc
   mapie.metrics.classification_ssc_score
   mapie.metrics.cumulative_differences
   mapie.metrics.expected_calibration_error
   mapie.metrics.hsic
   mapie.metrics.kolmogorov_smirnov_cdf
   mapie.metrics.kolmogorov_smirnov_p_value
   mapie.metrics.kolmogorov_smirnov_statistic
   mapie.metrics.kuiper_cdf
   mapie.metrics.kuiper_p_value
   mapie.metrics.kuiper_statistic
   mapie.metrics.length_scale
   mapie.metrics.regression_coverage_score
   mapie.metrics.regression_coverage_score_v2
   mapie.metrics.regression_mean_width_score
   mapie.metrics.regression_ssc
   mapie.metrics.regression_ssc_score
   mapie.metrics.spiegelhalter_p_value
   mapie.metrics.spiegelhalter_statistic
   mapie.metrics.top_label_ece

Conformity Scores (Regression)
==============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseRegressionScore
   mapie.conformity_scores.AbsoluteConformityScore
   mapie.conformity_scores.GammaConformityScore
   mapie.conformity_scores.ResidualNormalisedScore

Conformity Scores (Classification)
==================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseClassificationScore
   mapie.conformity_scores.NaiveConformityScore
   mapie.conformity_scores.LACConformityScore
   mapie.conformity_scores.APSConformityScore
   mapie.conformity_scores.RAPSConformityScore
   mapie.conformity_scores.TopKConformityScore

Resampling
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.subsample.BlockBootstrap
   mapie.subsample.Subsample

Mondrian
========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.mondrian.MondrianCP
