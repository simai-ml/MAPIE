from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from mapie._compatibility import np_quantile
from mapie._typing import ArrayLike, NDArray
from mapie.utils import (check_alpha_and_n_samples,
                         check_defined_variables_predict_cqr,
                         check_estimator_fit_predict, check_lower_upper_bounds,
                         check_null_weight, fit_estimator)

from .regression import MapieRegressor


class MapieQuantileRegressor(MapieRegressor):
    """
    This class implements the conformalized quantile regression strategy
    as proposed by Romano et al. (2019) to make conformal predictions.
    The only valid ``method`` is ``"quantile"`` and the only valid
    ``cv`` is ``"split"``.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``QuantileRegressor`` instance.

        By default ``"None"``.

    method: str
        Method to choose for prediction, in this case, the only valid method
        is the ``"quantile"`` method.

        By default ``"quantile"``.

    cv: Optional[str]
        The cross-validation strategy for computing conformity scores.
        In theory a split method is implemented as it is needed to provide
        both a training and calibration set.

        By default ``None``.

    alpha: float
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``0.1``.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_: RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: List[RegressorMixin]
        - [0]: Estimator with quantile value of alpha/2
        - [1]: Estimator with quantile value of 1 - alpha/2
        - [2]: Estimator with quantile value of 0.5

    conformity_scores_: NDArray of shape (n_samples_train, 3)
        Conformity scores between ``y_calib`` and ``y_pred``.

        - [:, 0]: for ``y_calib`` coming from prediction estimator
          with quantile of alpha/2
        - [:, 1]: for ``y_calib`` coming from prediction estimator
          with quantile of 1 - alpha/2
        - [:, 2]: maximum of those first two scores

    n_calib_samples: int
        Number of samples in the calibration dataset.

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieQuantileRegressor
    >>> X_train = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_train = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> X_calib = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_calib = np.array([5, 7, 9, 4, 8, 1, 5, 7.5, 9.5, 12])
    >>> mapie_reg = MapieQuantileRegressor().fit(
    ...     X_train,
    ...     y_train,
    ...     X_calib=X_calib,
    ...     y_calib=y_calib
    ... )
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(y_pis[:, :, 0])
    [[-8.16666667 19.        ]
     [-6.33333333 20.83333333]
     [-4.5        22.66666667]
     [-2.66666667 24.5       ]
     [-0.83333333 26.33333333]
     [ 1.         28.16666667]]
    >>> print(y_pred)
    [ 5.  7.  9. 11. 13. 15.]
    """
    valid_methods_ = ["quantile"]
    fit_attributes = [
        "estimators_",
        "conformity_scores_",
        "n_calib_samples",
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
        method: str = "quantile",
        cv: Optional[str] = None,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
        )
        self.cv = cv
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
            Can only be a float value between ``0.0`` and ``1.0``.
            Represent the risk level of the confidence interval.
            Lower alpha produce larger (more conservative) prediction
            intervals. Alpha is the complement of the target coverage level.

            By default ``0.1``.

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
            If the value of ``alpha`` is not between ``0.0`` and ``1.0``.
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
                    "Invalid alpha. Allowed values are between 0.0 and 1.0."
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
        have a ``fit`` and ``predict`` attribute, but also need to allow
        a quantile loss and therefore also setting a quantile value.
        Note that there is a ``TypedDict`` to check which methods allow for
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
            If the estimator implements ``fit`` or ``predict`` methods.

        ValueError
            We check if it's a known estimator that does quantile regression
            according to the dictionnary set quantile_estimator_params.
            This dictionnary will need to be updated with the latest new
            available estimators.

        ValueError
            The estimator does not have the ``"loss_name"`` in its parameters
            and therefore can not be used as an estimator.

        ValueError
            There is no quantile ``"loss_name"`` and therefore this estimator
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
                        f"{self.quantile_estimator_params.keys()} "
                        "Or, add your base model to"
                        + " ``quantile_estimator_params``."
                    )

    def _check_cv(
        self,
        cv: Optional[str] = None
    ) -> str:
        """
        Check if cv argument is ``None``, ``"split"`` or ``"prefit"``.

        Parameters
        ----------
        cv : Optional[str], optional
           cv to check, by default ``None``.

        Returns
        -------
        str
            cv itself or a default ``"split"``.

        Raises
        ------
        ValueError
            Raises an error if the cv is anything else but the method
            ``"split"`` or ``"prefit"``.
            Only the split method has been implemented.
        """
        if cv is None:
            return "split"
        if cv in ("split", "prefit"):
            return cv
        else:
            raise ValueError(
                "Invalid cv method, only valid method is ``split``."
            )

    def _check_calib_set(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]
    ]:
        """
        Check if a calibration set has already been defined, if not, then
        we define one using the ``train_test_split`` method.

        Parameters
        ----------
        Same definition of parameters as for the ``fit`` method.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]
            - [0]: ArrayLike of shape (n_samples_*(1-calib_size), n_features)
                X_train
            - [1]: ArrayLike of shape (n_samples_*(1-calib_size),)
                y_train
            - [2]: ArrayLike of shape (n_samples_*calib_size, n_features)
                X_calib
            - [3]: ArrayLike of shape (n_samples_*calib_size,)
                y_calib
            - [4]: ArrayLike of shape (n_samples_,)
                sample_weight_train
        """
        if X_calib is None or y_calib is None:
            if sample_weight is None:
                X_train, X_calib, y_train, y_calib = train_test_split(
                        X,
                        y,
                        test_size=calib_size,
                        random_state=random_state,
                        shuffle=shuffle,
                        stratify=stratify
                )
                sample_weight_train = sample_weight
            else:
                (
                        X_train,
                        X_calib,
                        y_train,
                        y_calib,
                        sample_weight_train,
                        _,
                ) = train_test_split(
                        X,
                        y,
                        sample_weight,
                        test_size=calib_size,
                        random_state=random_state,
                        shuffle=shuffle,
                        stratify=stratify
                )
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
        X_train, X_calib = cast(ArrayLike, X_train), cast(ArrayLike, X_calib)
        y_train, y_calib = cast(ArrayLike, y_train), cast(ArrayLike, y_calib)
        sample_weight_train = cast(ArrayLike, sample_weight_train)
        return X_train, y_train, X_calib, y_calib, sample_weight_train

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

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
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
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        X_calib: Optional[ArrayLike] of shape (n_calib_samples, n_features)
            Calibration data.

        y_calib: Optional[ArrayLike] of shape (n_calib_samples,)
            Calibration labels.

        calib_size: Optional[float]
            If ``X_calib`` and ``y_calib`` are not defined,
            then the calibration dataset is created with the split
            defined by ``calib_size``.

        random_state: Optional[Union[int, np.random.RandomState]], default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

            By default ``None``.

        shuffle: bool, default=True
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then stratify must be None.

            By default ``True``.

        stratify: array-like, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            If not ``None``, data is split in a stratified fashion, using this
            as the class labels.
            Read more in the :ref:`User Guide <stratification>`.

            By default ``None``.

        Returns
        -------
        MapieQuantileRegressor
             The model itself.
        """
        self.cv = self._check_cv(cast(str, self.cv))

        # Initialization
        self.estimators_: List[RegressorMixin] = []
        if self.cv == "prefit":
            estimator = cast(List, self.estimator)
            alpha = self._check_alpha(self.alpha)
            self._check_prefit_params(estimator)
            X_calib, y_calib = indexable(X, y)

            self.n_calib_samples = _num_samples(y_calib)
            y_calib_preds = np.full(
                shape=(3, self.n_calib_samples),
                fill_value=np.nan
            )
            for i, est in enumerate(estimator):
                self.estimators_.append(est)
                y_calib_preds[i] = est.predict(X_calib).ravel()
            self.single_estimator_ = self.estimators_[2]
        else:
            # Checks
            self._check_parameters()
            checked_estimator = self._check_estimator(self.estimator)
            alpha = self._check_alpha(self.alpha)
            X, y = indexable(X, y)
            random_state = check_random_state(random_state)
            results = self._check_calib_set(
                X,
                y,
                sample_weight,
                X_calib,
                y_calib,
                calib_size,
                random_state,
                shuffle,
                stratify,
            )
            X_train, y_train, X_calib, y_calib, sample_weight_train = results
            X_train, y_train = indexable(X_train, y_train)
            X_calib, y_calib = indexable(X_calib, y_calib)
            y_train, y_calib = _check_y(y_train), _check_y(y_calib)
            self.n_calib_samples = _num_samples(y_calib)
            check_alpha_and_n_samples(self.alpha, self.n_calib_samples)
            sample_weight_train, X_train, y_train = check_null_weight(
                sample_weight_train,
                X_train,
                y_train
            )
            y_train = cast(NDArray, y_train)

            y_calib_preds = np.full(
                shape=(3, self.n_calib_samples),
                fill_value=np.nan
            )

            if isinstance(checked_estimator, Pipeline):
                estimator = checked_estimator[-1]
            else:
                estimator = checked_estimator
            name_estimator = estimator.__class__.__name__
            alpha_name = self.quantile_estimator_params[
                name_estimator
            ]["alpha_name"]
            for i, alpha_ in enumerate(alpha):
                cloned_estimator_ = clone(checked_estimator)
                params = {alpha_name: alpha_}
                if isinstance(checked_estimator, Pipeline):
                    cloned_estimator_[-1].set_params(**params)
                else:
                    cloned_estimator_.set_params(**params)
                self.estimators_.append(fit_estimator(
                    cloned_estimator_, X_train, y_train, sample_weight_train
                ))
                y_calib_preds[i] = self.estimators_[-1].predict(X_calib)
            self.single_estimator_ = self.estimators_[2]

        self.conformity_scores_ = np.full(
                shape=(3, self.n_calib_samples),
                fill_value=np.nan
            )
        self.conformity_scores_[0] = y_calib_preds[0] - y_calib
        self.conformity_scores_[1] = y_calib - y_calib_preds[1]
        self.conformity_scores_[2] = np.max(
            [
                self.conformity_scores_[0],
                self.conformity_scores_[1]
            ], axis=0
        )
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        symmetry: Optional[bool] = True,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from the
        quantile regression at the alpha values: alpha/2, 1 - (alpha/2)
        while adding a constant based uppon their residuals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Ensemble has not been defined in predict and therefore should
            will not have any effects in this method.

        alpha: Optional[Union[float, Iterable[float]]]
            For ``MapieQuantileRegresor`` the alpha has to be defined
            directly in initial arguments of the class.

        symmetry: Optional[bool]
            Deciding factor to whether to find the quantile value for
            each residuals separatly or to use the maximum of the two
            combined.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
                - [:, 0, :]: Lower bound of the prediction interval.
                - [:, 1, :]: Upper bound of the prediction interval.
        """
        check_is_fitted(self, self.fit_attributes)
        check_defined_variables_predict_cqr(ensemble, alpha)
        alpha = self.alpha if symmetry else self.alpha/2
        check_alpha_and_n_samples(alpha, self.n_calib_samples)

        n = self.n_calib_samples
        q = (1 - (alpha)) * (1 + (1 / n))

        y_preds = np.full(
            shape=(3, _num_samples(X)),
            fill_value=np.nan,
            dtype=float,
        )
        for i, est in enumerate(self.estimators_):
            y_preds[i] = est.predict(X)
        if symmetry:
            quantile = np.full(
                2,
                np_quantile(
                    self.conformity_scores_[2], q, method="higher"
                )
            )
        else:
            quantile = np.array(
                [
                    np_quantile(
                        self.conformity_scores_[0], q, method="higher"
                    ),
                    np_quantile(
                        self.conformity_scores_[1], q, method="higher"
                    )
                ]
            )
        y_pred_low = y_preds[0][:, np.newaxis] - quantile[0]
        y_pred_up = y_preds[1][:, np.newaxis] + quantile[1]
        check_lower_upper_bounds(y_preds, y_pred_low, y_pred_up)
        return y_preds[2], np.stack([y_pred_low, y_pred_up], axis=1)
