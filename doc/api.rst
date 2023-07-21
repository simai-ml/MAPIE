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
   
   metrics.regression_coverage_score
   metrics.classification_coverage_score
   metrics.regression_mean_width_score
   metrics.classification_mean_width_score
   metrics.expected_calibration_error
   metrics.top_label_ece
   metrics.regression_coverage_score_v2
   metrics.classification_coverage_score_v2
   metrics.regression_ssc
   metrics.regression_ssc_score
   metrics.classification_ssc
   metrics.classification_ssc_score
   metrics.hsic

Conformity scores
=================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   conformity_scores.AbsoluteConformityScore
   conformity_scores.GammaConformityScore


Resampling
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   subsample.BlockBootstrap
   subsample.Subsample
