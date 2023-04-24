=======
History
=======

##### (##########)
------------------

* Add cross conformal option for quantile regression
* Add split conformal option for regression and classification
* Update check method for calibration

0.6.4 (2023-04-05)
------------------

* Fix runtime warning with RAPS method

0.6.3 (2023-03-23)
------------------

* Fix bug when labels do not start at 0

0.6.2 (2023-03-22)
------------------

* Make MapieClassifier a scikit-learn object
* Update documentation for MapieClassifier

0.6.1 (2023-01-31)
------------------

* Fix still existing bug for classification with very low scores

0.6.0 (2023-01-19)
------------------

* Add RCPS and CRC for multilabel-classifcation
* Add Top-Label calibration
* Fix bug for classification with very low scores

0.5.0 (2022-10-20)
------------------

* Add RAPS method for classification
* Add theoretical description for RAPS

0.4.2 (2022-09-02)
------------------

* Add tutorial for time series
* Convert existing tutorials in .py
* Add prefit method for CQR
* Add tutorial for CQR

0.4.1 (2022-06-27)
------------------

* Add `packaging` library in requirements
* Fix displaying problem in pypi

0.4.0 (2022-06-24)
------------------

* Relax and fix typing
* Add Split Conformal Quantile Regression
* Add EnbPI method for Time Series Regression
* Add EnbPI Documentation
* Add example with heteroscedastic data
* Add `ConformityScore` class that allows the user to define custom conformity scores

0.3.2 (2022-03-11)
------------------

* Refactorize unit tests
* Add "naive" and "top-k" methods in MapieClassifier
* Include J+aB method in regression tutorial
* Add MNIST example for classification
* Add cross-conformal for classification
* Add `notebooks` folder containing notebooks used for generating documentation tutorials
* Uniformize the use of matrix k_ and add an argument "ensemble" to method "predict" in regression.py
* Add replication of the Chen Xu's tutorial testing Jackknife+aB vs Jackknife+
* Add Jackknife+-after-Bootstrap documentation
* Improve scikit-learn pipelines compatibility

0.3.1 (2021-11-19)
------------------

* Add Jackknife+-after-Bootstrap method and add mean and median as aggregation functions
* Add "cumulative_score" method in MapieClassifier
* Allow image as input in MapieClassifier

0.3.0 (2021-09-10)
------------------

* Renaming estimators.py module to regression.py
* New classification.py module with MapieClassifier class, that estimates prediction sets from softmax score
* New set of unit tests for classification.py module
* Modification of the documentation architecture
* Split example gallery into separate regression and classification galleries
* Add first classification examples
* Add method classification_coverage_score in the module metrics.py
* Fixed code error for plotting of interval widths in tutorial of documentation
* Added missing import statements in tutorial of documentation
* Refactorize tests of `n_jobs` and `verbose` in `utils.py`

0.2.3 (2021-07-09)
------------------

* Inclusion in conda-forge with updated release checklist
* Add time series example
* Add epistemic uncertainty example
* Remove CicleCI redundancy with ReadTheDocs
* Remove Pep8speaks
* Include linting in CI/CD
* Use PyPa github actions for releases

0.2.2 (2021-06-10)
------------------

* Set alpha parameter as predict argument, with None as default value
* Switch to github actions for continuous integration of the code
* Add image explaining MAPIE internals on the README

0.2.1 (2021-06-04)
------------------

* Add `cv="prefit"` option
* Add sample_weight argument in fit method

0.2.0 (2021-05-21)
------------------

* Add n_jobs argument using joblib parallel processing
* Allow `cv` to take the value -1 equivalently to `LeaveOneOut()`
* Introduce the `cv` parameter to get closer to scikit-learn API
* Remove the `n_splits`, `shuffle` and `random_state` parameters
* Simplify the `method` parameter
* Fix typos in documentation and add methods descriptions in sphinx
* Accept alpha parameter as a list or np.ndarray. If alpha is an Iterable, `.predict()` returns a np.ndarray of shape (n_samples, 3, len(alpha)).

0.1.4 (2021-05-07)
------------------

* Move all alpha related operations to predict
* Assume default LinearRegression if estimator is None
* Improve documentation
* `return_pred` argument is now `ensemble` boolean

0.1.3 (2021-04-30)
------------------

* Update PyPi homepage
* Set up publication workflows as a github action
* Update issue and pull request templates
* Increase sklearn compatibility (coverage_score and unit tests)

0.1.2 (2021-04-27)
------------------

* First release on PyPi

0.1.1 (2021-04-27)
------------------

* First release on TestPyPi

0.1.0 (2021-04-27)
------------------

* Implement metrics.coverage
* Implement estimators.MapieRegressor
