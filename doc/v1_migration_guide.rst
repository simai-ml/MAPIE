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


2. Key parameter changes
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

``cv`` (includes ``groups``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``cv`` parameter manages the cross-validation configuration, accepting either an integer to indicate the number of data splits or a ``BaseCrossValidator`` object for custom data splitting.

- **v0.9**: The ``cv`` parameter was included in ``MapieRegressor``, where it handled cross-validation. The option ``cv="prefit"`` was available for models that were already pre-trained.
- **v1**: The ``cv`` parameter is now only present in ``CrossConformalRegressor``, with the ``prefit`` option removed. Additionally, the ``groups`` parameter was removed from the ``fit`` method, allowing groups to be directly passed to ``cv`` for processing.

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

``aggregation_method``
~~~~~~~~~~~~~~~~~~~~~~
The ``aggregation_method`` parameter defines how predictions from multiple conformal regressors are aggregated when making point predictions.

- **v0.9**: Previously, the ``agg_function`` parameter specified the aggregation method, allowing options such as the mean or median of predictions. This was applicable only when using ensemble methods by setting ``ensemble=True`` in the ``predict`` method.
- **v1**: The ``agg_function`` parameter has been renamed to ``aggregation_method`` for clarity. It now serves the same purpose in selecting an aggregation technique but is specified at prediction time rather than during class initialization. Additionally, the ``ensemble`` parameter has been removed, as ``aggregation_method`` is relevant only to the ``CrossConformalRegressor`` and ``JackknifeAfterBootstrapRegressor`` classes.

``Other parameters``
~~~~~~~~~~~~~~~~~~~~
No more parameters with incorrect ``None`` defaults.

- **v0.9**: Eg: ``estimator`` had a ``None`` default value, even though the actual default value is ``LinearRegression()``. This was the case for other parameters as well.
- **v1**: All parameters now have explicit defaults.

Some parameters' name have been improved for clarity:

- ``optimize_beta`` -> ``minimize_interval_width``
- ``symmetry``-> ``symmetric_intervals``


3. Method changes
-----------------

In MAPIE v1, the conformal prediction workflow is more streamlined and modular, with distinct methods for training, calibration, and prediction. The calibration process in v1 consists of four steps.

Step 1: Data splitting
~~~~~~~~~~~~~~~~~~~~~~
In v0.9, Data splitting is done within two-phase process. First, data ``(X, y)`` was divided into training ``(X_train, y_train)`` and test ``(X_test, y_test)`` sets using ``train_test_split`` from ``sklearn``. In the second phase, the split between training and calibration was either done manually or handled internally by ``MapieRegressor``.

In v1, a ``conf_split`` function has been introduced to split the data ``(X, y)`` into training ``(X_train, y_train)``, calibration ``(X_calib, y_calib)``, and test sets ``(X_test, y_test)``.

This new approach in v1 gives users more control over data splitting, making it easier to manage training, calibration, and testing phases explicitly.  The ``CrossConformalRegressor`` is an exception, where train/calibration splitting happens internally because cross-validation requires more granular control over data splits.

Step 2 & 3: Model training and calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In v0.9, the ``fit`` method handled both model training and calibration.

In v1.0: MAPIE separates between the training and calibration:

- ``.fit()`` method:
  - In v1, ``fit`` only trains the model on training data, without handling calibration.
  - Additional fitting parameters, like ``sample_weight``, should be included in ``fit_params``, keeping this method focused on training alone.

- ``.conformalize()`` method:
  - This new method performs calibration after fitting, using separate calibration data ``(X_calib, y_calib)``.
  - ``predict_params`` can be passed here, allowing independent control over calibration and prediction stages.

Step 4: Making predictions (``predict`` and ``predict_set`` methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In MAPIE v0.9, both point predictions and prediction intervals were produced through the ``predict`` method.

MAPIE v1 introduces two distinct methods for prediction:
- ``.predict_set()`` is dedicated to generating prediction intervals (i.e., lower and upper bounds), clearly separating interval predictions from point predictions.
- ``.predict()`` now focuses solely on producing point predictions.


4. Migration example: MAPIE v0.9 to MAPIE v1
--------------------------------------------

Below is a side-by-side example of code in MAPIE v0.9 and its equivalent in MAPIE v1 using the new modular classes and methods.

MAPIE v0.9 code
~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    from mapie.estimator import MapieRegressor
    from mapie.conformity_scores import GammaConformityScore
    from sklearn.model_selection import train_test_split

    # Step 1: Split data
    X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y, test_size=0.4)
    X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test, test_size=0.5)

    # Step 2: Train the model on the training set
    prefit_model = LinearRegression().fit(X_train, y_train)

    # Step 3: Initialize MapieRegressor with the prefit model and gamma conformity score
    v0 = MapieRegressor(
        estimator=prefit_model,
        cv="prefit",
        conformity_score=GammaConformityScore()
    )

    # Step 4: Fit MAPIE on the calibration set
    v0.fit(X_conf, y_conf)

    # Step 5: Make predictions with confidence intervals
    prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)[1][:, :, 0]
    prediction_points_v0 = v0.predict(X_test)

Equivalent MAPIE v1 code
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    from mapie.estimator import SplitConformalRegressor
    from mapie.utils import conf_split

    # Step 1: Split data with conf_split (returns X_train, y_train, X_conf, y_conf, X_test, y_test)
    X_train, y_train, X_conf, y_conf, X_test, y_test = conf_split(X, y)

    # Step 2: Train the model on the training set
    prefit_model = LinearRegression().fit(X_train, y_train)

    # Step 3: Initialize SplitConformalRegressor with the prefit model, gamma conformity score, and prefit option
    v1 = SplitConformalRegressor(
        estimator=prefit_model,
        confidence_level=0.9,       # equivalent to alpha=0.1 in v0.9
        conformity_score="gamma",
        prefit=True
    )

    # Step 4: Calibrate the model with the conformalize method on the calibration set
    v1.conformalize(X_conf, y_conf)

    # Step 5: Make predictions with confidence intervals
    prediction_intervals_v1 = v1.predict_set(X_test)
    prediction_points_v1 = v1.predict(X_test)


5. Additional migration examples
--------------------------------

We will provide further migration examples :

- **Prefit Models**: Using ``SplitConformalRegressor`` with ``prefit=True``
- **Non-Prefit Models**:

  - ``SplitConformalRegressor`` without ``prefit``
  - ``CrossConformalRegressor`` with ``fit_params`` (e.g., ``sample_weight``) and ``predict_params``
  - ``ConformalizedQuantileRegressor`` with ``symmetric_intervals=False``
  - ``JackknifeAfterBootstrapRegressor`` with custom configurations