from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from ._compatibility import np_nanquantile
from ._typing import ArrayLike, NDArray
from .aggregation_functions import aggregate_all
from .conformity_scores import ConformityScore
from .regression import MapieRegressor
from .utils import (check_alpha, check_alpha_and_n_samples,
                    check_conformity_score, check_cv,
                    check_defined_variables_predict_cqr,
                    check_estimator_fit_predict, check_lower_upper_bounds,
                    check_n_features_in, check_nan_in_aposteriori_prediction,
                    check_null_weight, fit_estimator)


class MapieQuantileRegressor(MapieRegressor):
    """
    This class implements the conformalized quantile regression strategy
    as proposed by Romano et al. (2019) to make conformal predictions.
    The valid ``method`` and ``cv`` are the same as for MapieRegressor.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with fit and predict methods), by default ``None``.
        If ``None``, estimator defaults to a ``QuantileRegressor`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set conformity scores,
        - "base", based on validation sets conformity scores,
        - "plus", based on validation conformity scores and
          testing predictions,
        - "minmax", based on validation conformity scores and
          testing predictions (min/max among cross-validation clones).

        By default ``"plus"``.

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing conformity scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to 1, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation),
          - ``subsample.Subsample`` object (bootstrap).
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is ignored.
          All data provided in the ``fit`` method is then used
          for computing conformity scores only.
          At prediction time, quantiles of these conformity scores are used
          to provide a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          conformity scores estimate are disjoint.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, it will be set to 0.1.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    agg_function : str
        Determines how to aggregate predictions from perturbed models, both at
        training and prediction time.

        If ``None``, it is ignored except if cv class is ``Subsample``,
        in which case an error is raised.
        If "mean" or "median", returns the mean or median of the predictions
        computed from the out-of-folds models.
        Note: if you plan to set the ``ensemble`` argument to ``True`` in the
        ``predict`` method, you have to specify an aggregation function.
        Otherwise an error would be raised.

        The Jackknife+ interval can be interpreted as an interval around the
        median prediction, and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        When the cross-validation strategy is Subsample (i.e. for the
        Jackknife+-after-Bootstrap method), this function is also used to
        aggregate the training set in-sample predictions.

        If cv is ``"prefit"`` or ``"split"``, ``agg_function`` is ignored.

        By default ``"mean"``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    conformity_score : Optional[ConformityScore]
        ConformityScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default None value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` conformity
          score
        - ConformityScore: any ``ConformityScore`` class

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets in cumulated_score.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    alpha: float
        Between 0 and 1.0, represents the risk level of the confidence
        interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``0.1``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_: RegressorMixin
        Estimator fitted on the whole training set.

    estimators_ : List[RegressorMixin]
        - [0]: Estimator with quantile value of alpha/2
        - [1]: Estimator with quantile value of 1 - alpha/2
        - [2]: Estimator with quantile value of 0.5

    conformity_scores_ : NDArray of shape (n_samples_train, 3)
        Conformity scores between ``y_calib`` and ``y_pred``:
            - [:, 0]: for y_calib coming from prediction estimator with
            quantile of alpha/2
            - [:, 1]: for y_calib coming from prediction estimator with
            quantile of 1 - alpha/2
            - [:, 2]: maximum of those first two scores

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.quantile_regression import MapieQuantileRegressor
    >>> X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    >>> y = np.array([5, 7.5, 9.5, 10.5, 12.5, 15, 16, 17.5, 18.5, 20, 21])
    >>> mapie_reg = MapieQuantileRegressor(alpha=0.25, random_state=42)
    >>> mapie_reg = mapie_reg.fit(X, y)
    >>> y_pred, y_pis = mapie_reg.predict(X, alpha=0.25)
    >>> print(y_pis[:, :, 0])
    [[ 5.          6.4       ]
     [ 6.5         8.        ]
     [ 8.          9.6       ]
     [ 9.5        11.2       ]
     [11.         12.9       ]
     [12.66666667 14.6       ]
     [14.33333333 16.3       ]
     [16.         18.        ]
     [17.66666667 19.7       ]
     [19.33333333 21.4       ]
     [21.         23.1       ]]
    >>> print(y_pred)
    [ 5.92857143  7.5         9.07142857 10.64285714 12.21428571 13.78571429
     15.35714286 16.92857143 18.5        20.07142857 21.64285714]
    """
    fit_attributes = [
        "single_estimator_",
        "single_estimator_alpha_",
        "estimators_",
        "k_",
        "conformity_scores_",
        "conformity_score_function_",
        "n_features_in_",
    ]

    quantile_estimator_params = {
        "GradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "alpha"
        },
        "QuantileRegressor": {
            "loss_name": "quantile",
            "alpha_name": "quantile"
        },
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "quantile"
        },
        "LGBMRegressor": {
            "loss_name": "objective",
            "alpha_name": "alpha"
        },
    }

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        method: str = "plus",
        alpha: float = 0.1,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            test_size=test_size,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            conformity_score=conformity_score,
            random_state=random_state
        )
        self.alpha = alpha

    def _check_alpha(
        self,
        alpha: float = 0.1,
    ) -> NDArray:
        """
        Perform several checks on the alpha value and changes it from
        a float to an ArrayLike.

        Parameters
        ----------
        alpha : float
            Can only be a float value between 0 and 1.0.
            Represent the risk level of the confidence interval.
            Lower alpha produce larger (more conservative) prediction
            intervals. Alpha is the complement of the target coverage level.
            By default 0.1

        Returns
        -------
        ArrayLike
            An ArrayLike of three values:

            - [0]: alpha value of alpha/2
            - [1]: alpha value of of 1 - alpha/2
            - [2]: alpha value of 0.5

        Raises
        ------
        ValueError
            If alpha is not a float.

        ValueError
            If the value of alpha is not between 0 and 1.0.
        """
        if self.cv == "prefit":
            warnings.warn(
                "WARNING: The alpha that is set needs to be the same"
                + " as the alpha of your prefitted model in the following"
                " order [alpha/2, 1 - alpha/2, 0.5]"
            )
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha >= 1.0)):
                raise ValueError(
                    "Invalid alpha. Allowed values are between 0 and 1.0."
                )
            else:
                alpha_np = np.array([alpha / 2, 1 - alpha / 2, 0.5])
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )
        return alpha_np

    def _check_estimator(
        self,
        estimator: Optional[Union[RegressorMixin, Pipeline]] = None,
    ) -> Union[RegressorMixin, Pipeline]:
        """
        Perform several checks on the estimator to check if it has
        all the required specifications to be used with this methodology.
        The estimators that can be used in MapieQuantileRegressor need to
        have a ``fit`` and ``predict``attribute, but also need to allow
        a quantile loss and therefore also setting a quantile value.
        Note that there is a TypedDict to check which methods allow for
        quantile regression.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``QuantileRegressor`` instance
            with ``solver`` set to "highs".

        Raises
        ------
        ValueError
            If the estimator fit or predict methods.

        ValueError
            We check if it's a known estimator that does quantile regression
            according to the dictionnary set quantile_estimator_params.
            This dictionnary will need to be updated with the latest new
            available estimators.

        ValueError
            The estimator does not have the "loss_name" in its parameters and
            therefore can not be used as an estimator.

        ValueError
            There is no quantile "loss_name" and therefore this estimator
            can not be used as a ``MapieQuantileRegressor``.

        ValueError
            The parameter to set the alpha value does not exist in this
            estimator and therefore we cannot use it.
        """
        if estimator is None:
            return QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )

        if check_cv(self.cv) == "prefit":
            self._check_prefit_params(estimator)
            return estimator

        check_estimator_fit_predict(estimator)
        if isinstance(estimator, Pipeline):
            self._check_estimator(estimator[-1])
            return estimator
        else:
            name_estimator = estimator.__class__.__name__
            if name_estimator == "QuantileRegressor":
                return estimator
            else:
                if name_estimator in self.quantile_estimator_params:
                    param_estimator = estimator.get_params()
                    loss_name, alpha_name = self.quantile_estimator_params[
                        name_estimator
                    ].values()
                    if loss_name in param_estimator:
                        if param_estimator[loss_name] != "quantile":
                            raise ValueError(
                                "You need to set the loss/objective argument"
                                + " of your base model to ``quantile``."
                            )
                        else:
                            if alpha_name in param_estimator:
                                return estimator
                            else:
                                raise ValueError(
                                    "The matching parameter `alpha_name` for"
                                    " estimator does not exist. "
                                    "Make sure you set it when initializing "
                                    "your estimator."
                                )
                    else:
                        raise ValueError(
                            "The matching parameter `loss_name` for"
                            + " estimator does not exist."
                        )
                else:
                    raise ValueError(
                        "The base model does not seem to be accepted"
                        + " by MapieQuantileRegressor. \n"
                        "Give a base model among: \n"
                        f"{self.quantile_estimator_params.keys()}"
                        "Or, add your base model to"
                        + " ``quantile_estimator_params``."
                    )

    def _check_prefit_params(
        self,
        estimator: List[Union[RegressorMixin, Pipeline]],
    ) -> None:
        """
        Check the parameters set for the specific case of prefit
        estimators.

        Parameters
        ----------
        estimator : List[Union[RegressorMixin, Pipeline]]
            List of three prefitted estimators that should have
            pre-defined quantile levels of alpha/2, 1 - alpha/2 and 0.5.

        Raises
        ------
        ValueError
            If a non-iterable variable is provided for estimator.

        ValueError
            If less or more than three models are defined.

        Warning
            If X and y are defined, then warning that they are not used.

        ValueError
            If the calibration set is not defined.

        Warning
            If the alpha is defined, warns the user that it must be set
            accordingly with the prefit estimators.
        """
        if isinstance(estimator, Iterable) is False:
            raise ValueError(
                "Estimator for prefit must be an iterable object."
            )
        if len(estimator) == 3:
            for est in estimator:
                check_estimator_fit_predict(est)
                check_is_fitted(est)
        else:
            raise ValueError(
                    "You need to have provided 3 different estimators, they"
                    " need to be preset with alpha values in the following"
                    " order [alpha/2, 1 - alpha/2, 0.5]."
                    )

    def _pred_multi_alpha(self, X: ArrayLike) -> NDArray:
        """
        TODO
        Return a prediction per train sample for each test sample, by
        aggregation with matrix  ``k_``.

        Parameters
        ----------
            X: NDArray of shape (n_samples_test, n_features)
                Input data

        Returns
        -------
            NDArray of shape (n_samples_test, n_samples_train)
        """
        y_pred_alpha_list = []
        for i, est_list in enumerate(self.estimators_):
            y_pred_multi = np.column_stack(
                [e.predict(X) for e in est_list]
            )
            # At this point, y_pred_multi is of shape
            # (n_samples_test, n_estimators_). The method
            # ``_aggregate_with_mask`` fits it to the right size
            # thanks to the shape of k_.
            y_pred_alpha_list.append(
                self._aggregate_with_mask(y_pred_multi, self.k_[i])
            )

        y_pred_alpha = np.array(y_pred_alpha_list)

        return y_pred_alpha

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieQuantileRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        All the clones of the estimators for different quantile values are
        stored in order alpha/2, 1 - alpha/2, 0.5 in the ``estimators_``
        attribute. Residuals for the first two estimators and the maximum
        of residuals among these residuals are stored in the
        ``conformity_scores_`` attribute.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        Returns
        -------
        MapieQuantileRegressor
             The model itself.
        """
        # Checks
        self._check_parameters()
        cv = check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        alpha = self._check_alpha(self.alpha)
        estimator = self._check_estimator(self.estimator)
        agg_function = self._check_agg_function(self.agg_function)
        X, y = indexable(X, y)
        y = _check_y(y)
        sample_weight = cast(Optional[NDArray], sample_weight)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.conformity_score_function_ = check_conformity_score(
            self.conformity_score
        )
        y = cast(NDArray, y)
        n_samples = _num_samples(y)
        check_alpha_and_n_samples(alpha, n_samples)
        cv = cast(BaseCrossValidator, cv)

        # Initialization
        self.single_estimator_alpha_: List[RegressorMixin] = []
        self.estimators_: List[List[RegressorMixin]] = []

        def clone_estimator(_estimator, _alpha):
            cloned_estimator_ = clone(_estimator)
            if isinstance(_estimator, Pipeline):
                alpha_name = self.quantile_estimator_params[
                    _estimator[-1].__class__.__name__
                ]["alpha_name"]
                _params = {alpha_name: _alpha}
                cloned_estimator_[-1].set_params(**_params)
            else:
                alpha_name = self.quantile_estimator_params[
                    _estimator.__class__.__name__
                ]["alpha_name"]
                _params = {alpha_name: _alpha}
                cloned_estimator_.set_params(**_params)
            return cloned_estimator_

        # Work
        y_pred = np.full(
            shape=(3, n_samples),
            fill_value=np.nan,
            dtype=float,
        )
        self.conformity_scores_ = np.full(
            shape=(3, n_samples),
            fill_value=np.nan,
            dtype=float,
        )
        if self.cv != "prefit":
            pred_matrix = np.full(
                shape=(3, n_samples, cv.get_n_splits(X, y)),
                fill_value=np.nan,
                dtype=float,
            )
            self.k_ = np.full(
                shape=(3, n_samples, cv.get_n_splits(X, y)),
                fill_value=np.nan,
                dtype=float,
            )
        else:
            self.k_ = np.full(
                shape=(3, n_samples, 1), fill_value=np.nan, dtype=float
            )

        for i, alpha_ in enumerate(alpha):
            if self.cv == "prefit":
                self.single_estimator_alpha_.append(estimator[i])
                y_pred[i] = self.single_estimator_alpha_[-1].predict(X)
            else:
                cloned_estimator_ = clone_estimator(estimator, alpha_)
                self.single_estimator_alpha_.append(fit_estimator(
                    cloned_estimator_, X, y, sample_weight
                ))

                if self.method == "naive":
                    y_pred[i] = self.single_estimator_alpha_[-1].predict(X)
                else:
                    outputs = Parallel(n_jobs=self.n_jobs,
                                       verbose=self.verbose)(
                        delayed(self._fit_and_predict_oof_model)(
                            clone_estimator(estimator, alpha_),
                            X,
                            y,
                            train_index,
                            val_index,
                            sample_weight,
                        )
                        for train_index, val_index in cv.split(X)
                    )
                    new_estimators, predictions, val_indices = map(
                        list, zip(*outputs)
                    )

                    self.estimators_.append(new_estimators)

                    for j, val_ind in enumerate(val_indices):
                        pred_matrix[i, val_ind, j] = np.array(
                            predictions[j], dtype=float
                        )
                        self.k_[i, val_ind, j] = 1
                    check_nan_in_aposteriori_prediction(pred_matrix[i])

                    y_pred[i] = aggregate_all(agg_function, pred_matrix[i])

            self.conformity_score_function_.sym = False
            self.conformity_scores_[i] = (
                self.conformity_score_function_.get_conformity_scores(
                    y,
                    y_pred[i]
                )
            )

        if isinstance(cv, ShuffleSplit):
            self.single_estimator_ = self.estimators_[2][0]
        else:
            self.single_estimator_ = self.single_estimator_alpha_[2]

        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Conformity scores from the training set and predictions
        from the model clones are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of conformity scores (naive and base methods),
        - quantiles of (predictions +/- conformity scores) (plus method),
        - quantiles of (max/min(predictions) +/- conformity scores)
          (minmax method).

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If False, predictions are those of the model trained on the whole
            training set.
            If True, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If cv is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represents the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is None.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples,) and (n_samples, 2, n_alpha) if alpha is not None.

            - [:, 0, :]: Lower bound of the prediction interval.
            - [:, 1, :]: Upper bound of the prediction interval.
        """
        # Checks
        check_is_fitted(self, self.fit_attributes)
        check_defined_variables_predict_cqr(ensemble, alpha)
        self._check_ensemble(ensemble)

        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)

        n_samples = len(self.conformity_scores_[-1])
        alpha_np = cast(NDArray, check_alpha(alpha))
        check_alpha_and_n_samples(alpha_np, n_samples)

        y_pred = np.full(
            shape=(3, _num_samples(X)),
            fill_value=np.nan,
            dtype=float,
        )
        for i, est in enumerate(self.single_estimator_alpha_):
            y_pred[i] = est.predict(X)

        if self.method in self.no_agg_methods_ \
                or self.cv in self.no_agg_cv_:
            y_pred_multi_low = y_pred[0, :, np.newaxis]
            y_pred_multi_up = y_pred[1, :, np.newaxis]
        else:
            y_pred_multi = self._pred_multi_alpha(X)

            if self.method == "minmax":
                y_pred_multi_low = np.min(
                    y_pred_multi[0], axis=1, keepdims=True
                )
                y_pred_multi_up = np.max(
                    y_pred_multi[1], axis=1, keepdims=True
                )
            else:
                y_pred_multi_low = y_pred_multi[0]
                y_pred_multi_up = y_pred_multi[1]

            if ensemble:
                for i, est in enumerate(self.single_estimator_alpha_):
                    y_pred[i] = aggregate_all(
                        self.agg_function, y_pred_multi[i]
                    )

        # compute distributions of lower and upper bounds
        conformity_scores_low = self.conformity_scores_[0]
        conformity_scores_up = self.conformity_scores_[1]
        alpha_np = alpha_np/2

        lower_bounds = (
            self.conformity_score_function_.get_estimation_distribution(
                y_pred_multi_low, conformity_scores_low
            )
        )
        upper_bounds = (
            self.conformity_score_function_.get_estimation_distribution(
                y_pred_multi_up, conformity_scores_up
            )
        )

        # get desired confidence intervals according to alpha
        y_pred_low = np.column_stack(
            [
                np_nanquantile(
                    lower_bounds.astype(float),
                    _alpha,
                    axis=1,
                    method="lower",
                )
                for _alpha in alpha_np
            ]
        )
        y_pred_up = np.column_stack(
            [
                np_nanquantile(
                    upper_bounds.astype(float),
                    1-_alpha,
                    axis=1,
                    method="higher",
                )
                for _alpha in alpha_np
            ]
        )

        for i in range(len(alpha_np)):
            check_lower_upper_bounds(y_pred, y_pred_low[:, i], y_pred_up[:, i])

        return y_pred[2], np.stack([y_pred_low, y_pred_up], axis=1)
