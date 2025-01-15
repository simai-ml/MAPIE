Migrating to MAPIE v1
===========================================

MAPIE v1 introduces several updates, enhancements, and structural changes that simplify the API by breaking down ``MapieRegressor`` functionality into dedicated classes for different conformal prediction methods. This guide outlines the key differences between MAPIE v0.9 and MAPIE v1 and provides instructions for adapting your code to the new structure.

1. Overview of class restructuring
-----------------------------------

MAPIE v1 organizes the ``MapieRegressor`` functionality into specific regressor classes, each optimized for a particular type of conformal prediction:

- ``SplitConformalRegressor``: Handles split conformal prediction.
- ``CrossConformalRegressor``: Implements cross-validation-based conformal prediction.
- ``JackknifeAfterBootstrapRegressor``: Supports jackknife-after-bootstrap conformal prediction.
- ``ConformalizedQuantileRegressor``: For quantile-based conformal prediction.

This modular approach makes it easier to select and configure a specific conformal regression method. Each class includes parameters relevant to its own methodology, reducing redundancy and improving readability.

Migration summary of ``MapieRegressor`` to new classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In MAPIE v0.9, ``MapieRegressor`` managed all conformal regression methods under a single interface, which sometimes led to parameter redundancy and ambiguity. In MAPIE v1, each method-specific class includes only the parameters and methods relevant to its method.

+--------------------+--------------------------------------------------------------------------+
| MAPIE v0.9 Class   | MAPIE v1 Classes                                                         |
+====================+==========================================================================+
| ``MapieRegressor`` | ``SplitConformalRegressor``, ``CrossConformalRegressor``,                |
|                    |                                                                          |
|                    | ``JackknifeAfterBootstrapRegressor``, ``ConformalizedQuantileRegressor`` |
+--------------------+--------------------------------------------------------------------------+


2. Method changes
-----------------

In MAPIE v1, the conformal prediction workflow is more streamlined and modular, with distinct methods for training, conformalization (named calibration in the scientific literature), and prediction. The conformalization process in v1 consists of four steps.

Step 1: Data splitting
~~~~~~~~~~~~~~~~~~~~~~
In v0.9, data splitting is handled by MAPIE.

In v1, the data splitting is left to the user, with the exception of cross-conformal methods (``CrossConformalRegressor``). The user can split the data into training, conformalization, and test sets using scikit-learn's ``train_test_split`` or other methods.

Step 2 & 3: Model training and conformalization (ie: calibration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In v0.9, the ``fit`` method handled both model training and calibration.

In v1.0: MAPIE separates between training and calibration. We decided to name the *calibration* step *conformalization*, to avoid confusion with probability calibration.

- ``.fit()`` method:
  - In v1, ``fit`` only trains the model on training data, without handling conformalization.
  - Additional fitting parameters, like ``sample_weight``, should be included in ``fit_params``, keeping this method focused on training alone.

- ``.conformalize()`` method:
  - This new method performs conformalization after fitting, using separate conformity data ``(X_conf, y_conf)``.
  - ``predict_params`` can be passed here, allowing independent control over conformalization and prediction stages.

Step 4: Making predictions (``predict`` and ``predict_set`` methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In MAPIE v0.9, both point predictions and prediction intervals were produced through the ``predict`` method.

MAPIE v1 introduces two distinct methods for prediction:
- ``.predict_set()`` is dedicated to generating prediction intervals (i.e., lower and upper bounds), clearly separating interval predictions from point predictions.
- ``.predict()`` now focuses solely on producing point predictions.



3. Key parameter changes
------------------------

``conformity_score``
~~~~~~~~~~~~~~~~~~~~
A parameter used to specify the scoring approach for evaluating model predictions.

- **v0.9**: Only allowed custom objects derived from ``BaseRegressionScore``.
- **v1**: Now accepts both strings (like ``"absolute"``) for predefined methods and custom ``BaseRegressionScore`` instances, simplifying usage.

``confidence_level``
~~~~~~~~~~~~~~~~~~~~
Indicates the desired coverage probability of the prediction intervals.

- **v0.9**: Specified as ``alpha`` during prediction, representing error rate.
- **v1**: Replaced with ``confidence_level`` to denote the coverage rate directly. Set at model initialization, improving consistency and clarity. ``confidence_level`` is equivalent to ``1 - alpha``.

``method``
~~~~~~~~~~
Specifies the approach for calculating prediction intervals, especially in advanced models like Cross Conformal and Jackknife After Bootstrap regressors.

- **v0.9**: Part of ``MapieRegressor``. Configured for the main prediction process.
- **v1**: Specific to ``CrossConformalRegressor`` and ``JackknifeAfterBootstrapRegressor``, indicating the interval calculation approach (``"base"``, ``"plus"``, or ``"minmax"``).

``cv``
~~~~~~~
The ``cv`` parameter manages the cross-validation configuration, accepting either an integer to indicate the number of data splits or a ``BaseCrossValidator`` object for custom data splitting.

- **v0.9**: The ``cv`` parameter was included in ``MapieRegressor``, where it handled cross-validation. The option ``cv="prefit"`` was available for models that were already pre-trained.
- **v1**: The ``cv`` parameter is now only present in ``CrossConformalRegressor``, with the ``prefit`` option removed.

``groups``
~~~~~~~~~~~
The ``groups`` parameter is used to specify group labels for cross-validation, ensuring that the same group is not present in both training and conformity sets.

- **v0.9**: Passed as a parameter to the ``fit`` method.
- **v1**: The ``groups`` present is now only present in ``CrossConformalRegressor``. It is passed in the ``.conformalize()`` method instead of the ``.fit()`` method. In other classes (like ``SplitConformalRegressor``), groups can be directly handled by the user during data splitting.

``prefit``
~~~~~~~~~~
Controls whether the model has been pre-fitted before applying conformal prediction.

- **v0.9**: Indicated through ``cv="prefit"`` in ``MapieRegressor``.
- **v1**: ``prefit`` is now a separate boolean parameter, allowing explicit control over whether the model has been pre-fitted before applying conformal methods.

``fit_params`` (includes ``sample_weight``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dictionary of parameters specifically used during training, such as ``sample_weight`` in scikit-learn.

- **v0.9**: Passed additional parameters in a flexible but less explicit manner.
- **v1**: Now explicitly structured as a dedicated dictionary, ``fit_params``, ensuring parameters used during training are clearly defined and separated from other stages.

``predict_params``
~~~~~~~~~~~~~~~~~~
Defines additional parameters exclusively for prediction.

- **v0.9**: Passed additional parameters in a flexible but less explicit manner, sometimes mixed within training configurations.
- **v1**: Now structured as a dedicated dictionary, ``predict_params``, to be used during calibration (``conformalize`` method) and prediction stages, ensuring no overlap with training parameters.

``agg_function``, ``aggregation_method``, ``aggregate_predictions``, and ``ensemble``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The aggregation method and technique for combining predictions in ensemble methods.

- **v0.9**: Previously, the ``agg_function`` parameter had two usage: to aggregate predictions when setting ``ensemble=True`` in the ``predict`` method, and to specify the aggregation technique in ``JackknifeAfterBootstrapRegressor``.
- **v1**: The ``agg_function`` parameter has been split into two distinct parameters: ``aggregate_predictions`` and ``aggregation_method``. ``aggregate_predictions`` is specific to ``CrossConformalRegressor``, and it specifies how predictions from multiple conformal regressors are aggregated when making point predictions. ``aggregation_method`` is specific to ``JackknifeAfterBootstrapRegressor``, and it specifies the aggregation technique for combining predictions across different bootstrap samples during conformalization.

``random_state``
~~~~~~~~~~~~~~~~~~

- **v0.9**: This parameter was used to control the randomness of the data splitting.
- **v1**: This parameter has been removed in cases where data splitting is now manual. Future evolutions may reintroduce it as a general purpose randomness control parameter.

``Other parameters``
~~~~~~~~~~~~~~~~~~~~
No more parameters with incorrect ``None`` defaults.

- **v0.9**: Eg: ``estimator`` had a ``None`` default value, even though the actual default value is ``LinearRegression()``. This was the case for other parameters as well.
- **v1**: All parameters now have explicit defaults.

Some parameters' name have been improved for clarity:

- ``optimize_beta`` -> ``minimize_interval_width``
- ``symmetry``-> ``symmetric_intervals``


4. Migration example: MAPIE v0.9 to MAPIE v1
----------------------------------------------------------------------------------------

Below is a side-by-side example of code in MAPIE v0.9 and its equivalent in MAPIE v1 using the new modular classes and methods.

Example 1: Split Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
############
Split conformal prediction is a widely used method for generating prediction intervals, it splits the data into training, conformity, and test sets. The model is trained on the training set, calibrated on the conformity set, and then used to make predictions on the test set. In `MAPIE v1`, the `SplitConformalRegressor` replaces the older `MapieRegressor` with a more modular design and simplified API.

MAPIE v0.9 Code
###############

Below is a MAPIE v0.9 code for split conformal prediction in case of pre-fitted model:

.. testcode::

    from sklearn.linear_model import LinearRegression
    from mapie.regression import MapieRegressor
    from mapie.conformity_scores import ResidualNormalisedScore
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y)
    X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test)

    prefit_model = LinearRegression().fit(X_train, y_train)

    v0 = MapieRegressor(
        estimator=prefit_model,
        cv="prefit",
        conformity_score=ResidualNormalisedScore()
    )

    v0.fit(X_conf, y_conf)

    prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)[1][:, :, 0]
    prediction_points_v0 = v0.predict(X_test)

Equivalent MAPIE v1 code
########################

Below is the equivalent MAPIE v1 code for split conformal prediction:

.. testcode::

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from mapie_v1.regression import SplitConformalRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y)
    X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test)

    prefit_model = LinearRegression().fit(X_train, y_train)

    v1 = SplitConformalRegressor(
        estimator=prefit_model,
        confidence_level=0.9,
        conformity_score="residual_normalized",
        prefit=True
    )

    # Here we're not using v1.fit(), because the provided model is already fitted
    v1.conformalize(X_conf, y_conf)

    prediction_intervals_v1 = v1.predict_set(X_test)
    prediction_points_v1 = v1.predict(X_test)

Example 2: Cross-Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
############

Cross-conformal prediction extends split conformal prediction by using multiple cross-validation folds to improve the efficiency of the prediction intervals. In MAPIE v1, `CrossConformalRegressor`` replaces the older `MapieRegressor`` for this purpose.

MAPIE v0.9 code
###############

Below is a MAPIE v0.9 code for cross-conformal prediction:

.. testcode::

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from mapie.regression import MapieRegressor
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.datasets import make_regression

    X_full, y_full = make_regression(n_samples=100, n_features=2, noise=0.1)
    X, X_test, y, y_test = train_test_split(X_full, y_full)
    groups = np.random.randint(0, 10, X.shape[0])
    sample_weight = np.random.rand(X.shape[0])

    regression_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5
    )

    v0 = MapieRegressor(
        estimator=regression_model,
        cv=GroupKFold(),
        agg_function="median",
    )

    v0.fit(X, y, sample_weight=sample_weight, groups=groups)

    prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)[1][:, :, 0]
    prediction_points_v0 = v0.predict(X_test, ensemble=True)

Equivalent MAPIE v1 code
########################

Below is the equivalent MAPIE v1 code for cross-conformal prediction:

.. testcode::

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GroupKFold
    from mapie_v1.regression import CrossConformalRegressor
    from sklearn.datasets import make_regression

    X_full, y_full = make_regression(n_samples=100, n_features=2, noise=0.1)
    X, X_test, y, y_test = train_test_split(X_full, y_full)
    groups = np.random.randint(0, 10, X.shape[0])
    sample_weight = np.random.rand(X.shape[0])

    regression_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5
    )

    v1 = CrossConformalRegressor(
        estimator=regression_model,
        confidence_level=0.9,
        cv=GroupKFold(),
        conformity_score="absolute",
    )

    v1.fit(X, y, fit_params={"sample_weight": sample_weight})
    v1.conformalize(X, y, groups=groups)

    prediction_intervals_v1 = v1.predict_set(X_test)
    prediction_points_v1 = v1.predict(X_test, aggregate_predictions="median")
