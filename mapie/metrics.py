from typing import Optional, cast, Union

import numpy as np
from sklearn.utils.validation import check_array, column_or_1d

from ._typing import ArrayLike, NDArray
from .utils import (calc_bins,
                    check_array_shape_classification,
                    check_array_shape_regression,
                    check_binary_zero_one,
                    check_nb_intervals_sizes,
                    check_nb_sets_sizes,
                    check_number_bins,
                    check_split_strategy)


def regression_coverage_score(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
) -> float:
    """
    Effective coverage score obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        True labels.
    y_pred_low: ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up: ArrayLike of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    float
        Effective coverage obtained by the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import regression_coverage_score
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(regression_coverage_score(y_true, y_pred_low, y_pred_up))
    0.8
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))
    coverage = np.mean(
        ((y_pred_low <= y_true) & (y_pred_up >= y_true))
    )
    return float(coverage)


def classification_coverage_score(
    y_true: ArrayLike,
    y_pred_set: ArrayLike
) -> float:
    """
    Effective coverage score obtained by the prediction sets.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction sets.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        True labels.
    y_pred_set: ArrayLike of shape (n_samples, n_class)
        Prediction sets given by booleans of labels.

    Returns
    -------
    float
        Effective coverage obtained by the prediction sets.

    Examples
    --------
    >>> from mapie.metrics import classification_coverage_score
    >>> import numpy as np
    >>> y_true = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True],
    ...     [False,  True,  True, False],
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True]
    ... ])
    >>> print(classification_coverage_score(y_true, y_pred_set))
    0.8
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    coverage = np.take_along_axis(
        y_pred_set, y_true.reshape(-1, 1), axis=1
    ).mean()
    return float(coverage)


def regression_mean_width_score(
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike
) -> float:
    """
    Effective mean width score obtained by the prediction intervals.

    Parameters
    ----------
    y_pred_low: ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up: ArrayLike of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    float
        Effective mean width of the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import regression_mean_width_score
    >>> import numpy as np
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(regression_mean_width_score(y_pred_low, y_pred_up))
    2.3
    """
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))
    mean_width = np.abs(y_pred_up - y_pred_low).mean()
    return float(mean_width)


def classification_mean_width_score(y_pred_set: ArrayLike) -> float:
    """
    Mean width of prediction set output by
    :class:`~mapie.classification.MapieClassifier`.

    Parameters
    ----------
    y_pred_set: ArrayLike of shape (n_samples, n_class)
        Prediction sets given by booleans of labels.

    Returns
    -------
    float
        Mean width of the prediction set.

    Examples
    --------
    >>> from mapie.metrics import classification_mean_width_score
    >>> import numpy as np
    >>> y_pred_set = np.array([
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True],
    ...     [False,  True,  True, False],
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True]
    ... ])
    >>> print(classification_mean_width_score(y_pred_set))
    2.0
    """
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    mean_width = y_pred_set.sum(axis=1).mean()
    return float(mean_width)


def expected_calibration_error(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
) -> float:
    """
    The expected calibration error, which is the difference between
    the confidence scores and accuracy per bin [1].

    [1] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
    "Obtaining well calibrated probabilities using bayesian binning."
    Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_score: ArrayLike of shape (n_samples,) or (n_samples, n_classes)
        The predictions scores.
    num_bins: int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy: str
        The way of splitting the predictions into different bins.
        The allowed split strategies are "uniform", "quantile" and
        "array split".
    Returns
    -------
    float
        The score of ECE (Expected Calibration Error).
    """
    split_strategy = check_split_strategy(split_strategy)
    num_bins = check_number_bins(num_bins)
    y_true_ = check_binary_zero_one(y_true)
    y_scores = cast(NDArray, y_scores)

    if np.size(y_scores.shape) == 2:
        y_score = cast(
            NDArray, column_or_1d(np.nanmax(y_scores, axis=1))
        )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))

    _, bin_accs, bin_confs, bin_sizes = calc_bins(
        y_true_, y_score, num_bins, split_strategy
    )

    return np.divide(
        np.sum(bin_sizes * np.abs(bin_accs - bin_confs)),
        np.sum(bin_sizes)
    )


def top_label_ece(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    y_score_arg: Optional[ArrayLike] = None,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
    classes: Optional[ArrayLike] = None,
) -> float:
    """
    The Top-Label ECE which is a method adapted to fit the
    ECE to a Top-Label setting [2].

    [2] Gupta, Chirag, and Aaditya K. Ramdas.
    "Top-label calibration and multiclass-to-binary reductions."
    arXiv preprint arXiv:2107.08353 (2021).

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_scores: ArrayLike of shape (n_samples, n_classes)
    or (n_samples,)
        The predictions scores, either the maximum score and the
        argmax needs to be inputted or in the form of the prediction
        probabilities.
    y_score_arg: Optional[ArrayLike] of shape (n_samples,)
        If only the maximum is provided in the y_scores, the argmax must
        be provided here. This is optional and could be directly infered
        from the y_scores.
    num_bins: int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy: str
        The way of splitting the predictions into different bins.
        The allowed split strategies are "uniform", "quantile" and
        "array split".
    classes: ArrayLike of shape (n_samples,)
        The different classes, in order of the indices that would be
        present in a pred_proba.

    Returns
    -------
    float
        The ECE score adapted in the top label setting.
    """
    ece = float(0.)
    split_strategy = check_split_strategy(split_strategy)
    num_bins = check_number_bins(num_bins)
    y_true = cast(NDArray, column_or_1d(y_true))
    if y_score_arg is None:
        y_score = cast(
            NDArray, column_or_1d(np.nanmax(y_scores, axis=1))
        )
        if classes is None:
            y_score_arg = cast(
                NDArray, column_or_1d(np.nanargmax(y_scores, axis=1))
            )
        else:
            classes = cast(NDArray, classes)
            y_score_arg = cast(
                NDArray, column_or_1d(classes[np.nanargmax(y_scores, axis=1)])
            )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))
        y_score_arg = cast(NDArray, column_or_1d(y_score_arg))
    labels = np.unique(y_score_arg)

    for label in labels:
        label_ind = np.where(label == y_score_arg)[0]
        y_true_ = np.array(y_true[label_ind] == label, dtype=int)
        ece += expected_calibration_error(
            y_true_,
            y_scores=y_score[label_ind],
            num_bins=num_bins,
            split_strategy=split_strategy
        )
    ece /= len(labels)
    return ece


def regression_coverage_score_v2(
    y_true: NDArray,
    y_intervals: NDArray,
) -> NDArray:
    """
    Effective coverage score obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    It is different from ``regression_coverage_score`` because it uses
    directly the output of ``predict`` method and can compute the
    coverage for each alpha.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples, n_alpha) or (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha)
        Lower and upper bound of prediction intervals
        with different alpha risks.

    Returns
    -------
    NDArray of shape (n_alpha,)
        Effective coverage obtained by the prediction intervals.
    """
    y_intervals = check_array_shape_regression(y_true, y_intervals)
    if len(y_true.shape) != 2:
        y_true = cast(NDArray, column_or_1d(y_true))
        y_true = np.expand_dims(y_true, axis=1)
    coverages = np.mean(
        np.logical_and(
            np.less_equal(y_intervals[:, 0, :], y_true),
            np.greater_equal(y_intervals[:, 1, :], y_true)
        ),
        axis=0
    )
    return coverages


def classification_coverage_score_v2(
    y_true: NDArray,
    y_pred_set: NDArray
) -> NDArray:
    """
    Effective coverage score obtained by the prediction sets.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction sets.

    It is different from ``classification_coverage_score`` because it uses
    directly the output of ``predict`` method and can compute the
    coverage for each alpha.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples, n_alpha) or (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
        Prediction sets given by booleans of labels.

    Returns
    -------
    NDArray of shape (n_alpha,)
        Effective coverage obtained by the prediction sets.
    """
    y_pred_set = check_array_shape_classification(y_true, y_pred_set)
    if len(y_true.shape) != 2:
        y_true = cast(NDArray, column_or_1d(y_true))
        y_true = np.expand_dims(y_true, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    coverage = np.nanmean(
        np.take_along_axis(y_pred_set, y_true, axis=1),
        axis=0
    )
    return coverage[0]


def regression_ssc(
    y_true: NDArray,
    y_intervals: NDArray,
    num_bins: int = 3
) -> NDArray:
    """
    Compute Size-Stratified Coverage metrics proposed in [3] that is
    the conditional coverage conditioned by the size of the intervals.
    The intervals are ranked by their size (ascending) and then divided into
    num_bins groups: one value of coverage by groups is computed.

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    [3] Angelopoulos, A. N., & Bates, S. (2021).
    A gentle introduction to conformal prediction and
    distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_alpha, num_bins)

    Examples
    --------
    >>> from mapie.metrics import regression_ssc
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5])
    >>> y_intervals = np.array([
    ... [4, 6],
    ... [6.0, 9.0],
    ... [9, 10.0]
    ... ])
    >>> print(regression_ssc(y_true, y_intervals, num_bins=2))
    [[1. 1.]]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_intervals = check_array_shape_regression(y_true, y_intervals)
    check_number_bins(num_bins)
    widths = np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :])
    check_nb_intervals_sizes(widths, num_bins)

    indexes_sorted = np.argsort(widths, axis=0)
    indexes_bybins = np.array_split(indexes_sorted, num_bins, axis=0)
    coverages = np.zeros((y_intervals.shape[2], num_bins))
    for i, indexes in enumerate(indexes_bybins):
        intervals_binned = np.stack([
            np.take_along_axis(y_intervals[:, 0, :], indexes, axis=0),
            np.take_along_axis(y_intervals[:, 1, :], indexes, axis=0)
        ], axis=1)
        coverages[:, i] = regression_coverage_score_v2(y_true[indexes],
                                                       intervals_binned)

    return coverages


def regression_ssc_score(
    y_true: NDArray,
    y_intervals: NDArray,
    num_bins: int = 3
) -> NDArray:
    """
    Aggregate by the minimum for each alpha the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_alpha,)

    Examples
    --------
    >>> from mapie.metrics import regression_ssc
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5])
    >>> y_intervals = np.array([
    ... [[4, 4], [6, 7.5]],
    ... [[6.0, 8], [9.0, 10]],
    ... [[9, 9], [10.0, 10.0]]
    ... ])
    >>> print(regression_ssc_score(y_true, y_intervals, num_bins=2))
    [1.  0.5]
    """
    return np.min(regression_ssc(y_true, y_intervals, num_bins), axis=1)


def classification_ssc(
    y_true: NDArray,
    y_pred_set: NDArray,
    num_bins: Union[int, None] = None
) -> NDArray:
    """
    Compute Size-Stratified Coverage metrics proposed in [3] that is
    the conditional coverage conditioned by the size of the predictions sets.
    The sets are ranked by their size (ascending) and then divided into
    num_bins groups: one value of coverage by groups is computed.

    [3] Angelopoulos, A. N., & Bates, S. (2021).
    A gentle introduction to conformal prediction and
    distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than the
        number of different set sizes.

    Returns
    -------
    NDArray of shape (n_alpha, num_bins)

    Examples
    --------
    >>> from mapie.metrics import classification_ssc
    >>> import numpy as np
    >>> y_true = y_true_class = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...    [True, True, True, True],
    ...    [False, True, False, True],
    ...    [True, True, True, False],
    ...    [False, False, True, True],
    ...    [True, True, False, True]])
    >>> print(classification_ssc(y_true, y_pred_set, num_bins=2))
    [[1.         0.66666667]]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_set = check_array_shape_classification(y_true, y_pred_set)

    sizes = np.sum(y_pred_set, axis=1)
    n_classes = y_pred_set.shape[1]
    if num_bins is None:
        bins = list(range(n_classes + 1))
    else:
        check_nb_sets_sizes(sizes, num_bins)
        check_number_bins(num_bins)
        bins = [
            b[0] for b in np.array_split(range(n_classes + 1), num_bins)
        ]

    digitized_sizes = np.digitize(sizes, bins)
    coverages = np.zeros((y_pred_set.shape[2], len(bins)))
    for alpha in range(y_pred_set.shape[2]):
        indexes_bybins = [
            np.argwhere(digitized_sizes[:, alpha] == i)
            for i in range(1, len(bins)+1)
        ]

        for i, indexes in enumerate(indexes_bybins):
            coverages[alpha, i] = classification_coverage_score_v2(
                y_true[indexes],
                np.take_along_axis(
                    y_pred_set[:, :, alpha],
                    indexes,
                    axis=0
                )
            )
    return coverages


def classification_ssc_score(
    y_true: NDArray,
    y_pred_set: NDArray,
    num_bins: Union[int, None] = None
) -> NDArray:
    """
    Aggregate by the minimum for each alpha the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than
        the number of different set sizes.

    Returns
    -------
    NDArray of shape (n_alpha,)

    Examples
    --------
    >>> from mapie.metrics import classification_ssc_score
    >>> import numpy as np
    >>> y_true = y_true_class = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...    [True, True, True, True],
    ...    [False, True, False, True],
    ...    [True, True, True, False],
    ...    [False, False, True, True],
    ...    [True, True, False, True]])
    >>> print(classification_ssc_score(y_true, y_pred_set, num_bins=2))
    [0.66666667]
    """
    return np.nanmin(classification_ssc(y_true, y_pred_set, num_bins), axis=1)


def _gaussian_kernel(
    x: NDArray,
    kernel_size: int
) -> NDArray:
    """
    Computes the gaussian kernel of x. (Used in hsic function)

    Parameters
    ----------
    x: NDArray
        The values from which to compute the gaussian kernel.
    kernel_size: int
        The variance (sigma), this coefficient controls the width of the curve.
    """
    norm_x = x ** 2
    dist = -2 * np.matmul(x, x.transpose((0, 2, 1))) \
        + norm_x + norm_x.transpose((0, 2, 1))
    return np.exp(-dist / kernel_size)


def hsic(
    y_true: NDArray,
    y_intervals: NDArray,
    kernel_sizes: ArrayLike = (1, 1)
) -> NDArray:
    """
    Compute the square root of the hsic coefficient. HSIC is Hilbert-Schmidt
    independence criterion that is a correlation measure. Here we use it as
    proposed in [4], to compute the correlation between the indicator of
    coverage and the interval size.

    If hsic is 0, the two variables (the indicator of coverage and the
    interval size) are independant.

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    [4] Feldman, S., Bates, S., & Romano, Y. (2021).
    Improving conditional coverage via orthogonal quantile regression.
    Advances in Neural Information Processing Systems, 34, 2060-2071.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction sets given by booleans of labels.
    kernel_sizes: ArrayLike of size (2,)
        The variance (sigma) for each variable (the indicator of coverage and
        the interval size), this coefficient controls the width of the curve.

    Returns
    -------
    NDArray of shape (n_alpha,)
        One hsic correlation coefficient by alpha.

    Raises
    ------
    ValueError
        If kernel_sizes has a length different from 2
        and if it has negative or null values.

    Examples
    -------
    >>> from mapie.metrics import hsic
    >>> import numpy as np
    >>> y_true = np.array([9.5, 10.5, 12.5])
    >>> y_intervals = np.array([
    ... [[9, 9], [10.0, 10.0]],
    ... [[8.5, 9], [12.5, 12]],
    ... [[10.5, 10.5], [12.0, 12]]
    ... ])
    >>> print(hsic(y_true, y_intervals))
    [0.31787614 0.2962914 ]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_intervals = check_array_shape_regression(y_true, y_intervals)
    kernel_sizes = cast(NDArray, column_or_1d(kernel_sizes))
    if len(kernel_sizes) != 2:
        raise ValueError(
            "kernel_sizes should be an ArrayLike of length 2"
        )
    if (kernel_sizes <= 0).any():
        raise ValueError(
            "kernel_size should be positive"
        )
    n_samples, _, n_alpha = y_intervals.shape
    y_true_per_alpha = np.tile(y_true, (n_alpha, 1)).transpose()
    widths = np.expand_dims(
        np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :]).transpose(),
        axis=2
    )
    cov_ind = np.expand_dims(
        np.int_(
            ((y_intervals[:, 0, :] <= y_true_per_alpha) &
             (y_intervals[:, 1, :] >= y_true_per_alpha))
        ).transpose(),
        axis=2
    )

    k_mat = _gaussian_kernel(widths, kernel_sizes[0])
    l_mat = _gaussian_kernel(cov_ind, kernel_sizes[1])
    h_mat = np.eye(n_samples) - 1 / n_samples * np.ones((n_samples, n_samples))
    hsic_mat = np.matmul(l_mat, np.matmul(h_mat, np.matmul(k_mat, h_mat)))
    hsic_mat /= ((n_samples - 1) ** 2)
    coef_hsic = np.sqrt(np.matrix.trace(hsic_mat, axis1=1, axis2=2))

    return coef_hsic
