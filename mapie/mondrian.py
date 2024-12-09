from __future__ import annotations

from copy import copy
from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from mapie.calibration import MapieCalibrator
from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    AbsoluteConformityScore,
    APSConformityScore,
    GammaConformityScore,
    LACConformityScore,
    NaiveConformityScore,
    TopKConformityScore
)
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.regression import (
    MapieQuantileRegressor,
    MapieRegressor,
    MapieTimeSeriesRegressor
)
from mapie.utils import check_alpha
from mapie._typing import ArrayLike, NDArray


class MondrianCP(BaseEstimator):
    """Mondrian is a method for making conformal predictions
    for partition of individuals.

    The Mondrian method is implemented in the `MondrianCP` class. It takes as
    input a `MapieClassifier` or `MapieRegressor` estimator and fits a model
    for each group of individuals. The `MondrianCP` class can then be used to
    run a conformal prediction procedure for each of these groups and hence
    achieve marginal coverage on each of them.

    The underlying estimator must be used with `cv='prefit'` and the
    conformity score must be one of the following:
    - For `MapieClassifier`: 'lac', 'score', 'cumulated_score', 'aps' or 'topk'
    - For `MapieRegressor`: 'absolute' or 'gamma'

    Parameters
    ----------
    mapie_estimator : Union[MapieClassifier, MapieRegressor]
        The estimator for which the Mondrian method will be applied.
        It must be used with `cv='prefit'` and the
        conformity score must be one of the following:
        - For `MapieClassifier`: 'lac', 'score', 'cumulated_score', 'aps' or
        'topk'
        - For `MapieRegressor`: 'absolute' or 'gamma'

    Attributes
    ----------
    partition_groups : NDArray
        The unique groups of individuals for which the estimator was fitted

    mapie_estimators : Dict
        A dictionary containing the fitted conformal estimator for each group
        of individuals

    References
    ----------
    Vladimir Vovk, David Lindsay, Ilia Nouretdinov, and Alex Gammerman.
    Mondrian confidence machine.
    Technical report, Royal Holloway University of London, 2003

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mapie.classification import MapieClassifier
    >>> from mapie.mondrian import MondrianCP
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> partition_toy = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> clf = LogisticRegression(random_state=42).fit(X_toy, y_toy)
    >>> mapie = MondrianCP(MapieClassifier(estimator=clf, cv="prefit")).fit(
    ...     X_toy, y_toy, partition=partition_toy
    ... )
    >>> _, y_pi_mapie = mapie.predict(
    ...     X_toy, partition=partition_toy, alpha=0.4)
    >>> print(y_pi_mapie[:, :, 0].astype(bool))
    [[ True False False]
     [ True False False]
     [ True  True False]
     [ True  True False]
     [False  True False]
     [False  True  True]
     [False False  True]
     [False False  True]
     [False False  True]]
    """

    not_allowed_estimators = (
        MapieCalibrator,
        MapieMultiLabelClassifier,
        MapieQuantileRegressor,
        MapieTimeSeriesRegressor
    )
    allowed_classification_ncs_str = [
        "lac", "score", "cumulated_score", "aps", "top_k"
    ]
    allowed_classification_ncs_class = (
        LACConformityScore, NaiveConformityScore, APSConformityScore,
        TopKConformityScore
    )
    allowed_regression_ncs = (
        AbsoluteConformityScore, GammaConformityScore
    )
    fit_attributes = [
        "partition_groups",
        "mapie_estimators"
    ]

    def __init__(
        self,
        mapie_estimator: Union[MapieClassifier, MapieRegressor]
    ):
        self.mapie_estimator = mapie_estimator

    def fit(
        self, X: ArrayLike,
        y: ArrayLike,
        partition: ArrayLike,
        **fit_params
    ) -> MondrianCP:
        """
        Fit the Mondrian method

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data

        y : ArrayLike of shape (n_samples,) or (n_samples, n_outputs)
            The target values

        partition : ArrayLike of shape (n_samples,)
            The groups of individuals. Must be defined by integers. There must
            be at least 2 individuals per group.

        **fit_params
            Additional keyword arguments to pass to the estimator's fit method
            that may be specific to the Mapie estimator used
        """

        X, y, partition = self._check_fit_parameters(X, y, partition)
        self.partition_groups = np.unique(partition)
        self.mapie_estimators = {}

        if isinstance(self.mapie_estimator, MapieClassifier):
            self.n_classes = len(np.unique(y))

        for group in self.partition_groups:
            mapie_group_estimator = copy(self.mapie_estimator)
            indices_groups = np.argwhere(partition == group)[:, 0]
            X_g = [X[index] for index in indices_groups]
            y_g = [y[index] for index in indices_groups]
            mapie_group_estimator.fit(X_g, y_g, **fit_params)
            self.mapie_estimators[group] = mapie_group_estimator

        return self

    def predict(
        self,
        X: ArrayLike,
        partition: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        **predict_params
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Perform conformal prediction for each group of individuals

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data

        partition : ArrayLike of shape (n_samples,), optional
            The groups of individuals. Must be defined by integers.

            By default None.

        alpha : float or Iterable[float], optional
            The desired coverage level(s) for each group.

            By default None.

        **predict_params
            Additional keyword arguments to pass to the estimator's predict
            method that may be specific to the Mapie estimator used

        Returns
        -------
        y_pred : NDArray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values

        y_pss : NDArray of shape (n_samples, n_outputs, n_alpha)
            The predicted sets for the desired levels of coverage
        """
        check_is_fitted(self, self.fit_attributes)
        X = cast(NDArray, X)
        alpha_np = cast(NDArray, check_alpha(alpha))

        if alpha_np is None and self.mapie_estimator.estimator is not None:
            return self.mapie_estimator.estimator.predict(
                X, **predict_params
            )

        if isinstance(self.mapie_estimator, MapieClassifier):
            y_pred = np.empty((len(X), ))
            y_pss = np.empty((len(X), self.n_classes, len(alpha_np)))
        else:
            y_pred = np.empty((len(X),))
            y_pss = np.empty((len(X), 2, len(alpha_np)))

        partition = self._check_partition_predict(X, partition)
        partition_groups = np.unique(partition)

        for _, group in enumerate(partition_groups):
            indices_groups = np.argwhere(partition == group)[:, 0]
            X_g = [X[index] for index in indices_groups]
            y_pred_g, y_pss_g = self.mapie_estimators[group].predict(
                X_g, alpha=alpha_np, **predict_params
            )
            y_pred[indices_groups] = y_pred_g
            y_pss[indices_groups] = y_pss_g

        return y_pred, y_pss

    def _check_cv(self):
        """
        Check that the underlying Mapie estimator uses cv='prefit'

        Raises
        ------
        ValueError
            If the underlying Mapie estimator does not use cv='prefit'
        """
        if not self.mapie_estimator.cv == "prefit":
            raise ValueError(
                "Mondrian can only be used if the underlying" +
                "estimator is used in prefit mode."
            )

    def _check_partition_fit(self, X: NDArray, partition: NDArray):
        """
        Check that each group is defined by an integer and check that there
        are at least 2 individuals per group

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            The input data

        partition : NDArray of shape (n_samples,)

        Raises
        ------
        ValueError
            If the partition is not defined by integers
            If there is less than 2 individuals per group
            If the number of individuals in the partition is not equal to the
            number of rows in X
        """
        if not np.issubdtype(partition.dtype, np.integer):
            raise ValueError("The partition must be defined by integers")

        _, counts = np.unique(partition, return_counts=True)
        if np.min(counts) < 2:
            raise ValueError("There must be at least 2 individuals per group")

        self._check_partition_length(X, partition)

    def _check_partition_predict(
            self,
            X: NDArray,
            partition: ArrayLike
    ) -> NDArray:
        """
        Check that there is no new group in the prediction and that
        the number of individuals in the partition is equal to the number of
        rows in X

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            The input data

        partition : ArrayLike of shape (n_samples,)
            The groups of individuals. Must be defined by integers

        Returns
        -------
        partition : NDArray of shape (n_samples,)
            Partition of the dataset

        Raises
        ------
        ValueError
            If there is a new group in the prediction
        """
        partition = cast(NDArray, np.array(partition))
        if not np.all(np.isin(partition, self.partition_groups)):
            raise ValueError(
                "There is at least one new group in the prediction."
            )
        self._check_partition_length(X, partition)

        return partition

    def _check_partition_length(self, X: NDArray, partition: NDArray):
        """
        Check that the number of rows in the groups array is equal to
        the number of rows in the attributes array.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            The individual data.

        partition : NDArray of shape (n_samples,)
            The groups of individuals. Must be defined by integers

        Raises
        ------
        ValueError
            If the number of individuals in the partition is not equal to the
            number of rows in X
        """
        if len(partition) != len(X):
            raise ValueError(
                "The number of individuals in the partition must "
                "be equal to the number of rows in X"
            )

    def _check_estimator(self):
        """
        Check that the estimator is not in the `not_allowed_estimators`.

        Raises
        ------
        ValueError
            If the estimator is in the `not_allowed_estimators`.
        """
        if isinstance(self.mapie_estimator, self.not_allowed_estimators):
            raise ValueError(
                "The estimator must be a classifier or regressor"
            )

    def _check_confomity_score(self):
        """
        Check that the conformity score is in `allowed_classification_ncs_str`
        or `allowed_classification_ncs_class` if the estimator is a
        `MapieClassifier` or in the `allowed_regression_ncs` if the estimator
        is a `MapieRegressor`

        Raises
        ------
        ValueError
            If conformity score is not in the `allowed_classification_ncs_str`
            or `allowed_classification_ncs_class` if the estimator is a
            `MapieClassifier` or in the `allowed_regression_ncs` if the
            estimator is a `MapieRegressor`.
        """
        if isinstance(self.mapie_estimator, MapieClassifier):
            if self.mapie_estimator.method is not None:
                if self.mapie_estimator.method not in \
                   self.allowed_classification_ncs_str:
                    raise ValueError(
                        "The conformity score for the classifier must " +
                        f"be one of {self.allowed_classification_ncs_str}"
                    )

            if self.mapie_estimator.conformity_score is not None:
                if type(self.mapie_estimator.conformity_score) not in \
                   self.allowed_classification_ncs_class:
                    raise ValueError(
                        "The conformity score for the classifier must" +
                        f" be one of {self.allowed_classification_ncs_class}"
                    )
        else:
            if self.mapie_estimator.conformity_score is not None:
                if not isinstance(self.mapie_estimator.conformity_score,
                   self.allowed_regression_ncs):
                    raise ValueError(
                        "The conformity score for the regressor must " +
                        f"be one of {self.allowed_regression_ncs}"
                    )

    def _check_fit_parameters(
        self, X: ArrayLike, y: ArrayLike, partition: ArrayLike
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Perform checks on the input data, partition and the estimator

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data

        y : ArrayLike of shape (n_samples,) or (n_samples, n_outputs)
            The target values

        partition : ArrayLike of shape (n_samples,)
            The groups of individuals. Must be defined by integers

        Returns
        -------
        X : NDArray of shape (n_samples, n_features)
            The input data

        y : NDArray of shape (n_samples,) or (n_samples, n_outputs)
            The target values

        partition : NDArray of shape (n_samples,)
            The group values
        """
        self._check_estimator()
        self._check_cv()
        self._check_confomity_score()

        X, y = indexable(X, y)
        y = _check_y(y)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        partition = cast(NDArray, np.array(partition))

        self._check_partition_fit(X, partition)
        self._check_partition_length(X, partition)

        return X, y, partition
