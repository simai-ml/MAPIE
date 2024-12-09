from typing import Any, Optional, cast

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie._typing import NDArray
from mapie.classification import MapieClassifier
from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.sets import (
    APSConformityScore, LACConformityScore, NaiveConformityScore,
    RAPSConformityScore, TopKConformityScore
)
from mapie.conformity_scores.utils import check_classification_conformity_score
from mapie.utils import check_alpha


random_state = 42

cs_list = [
    None, LACConformityScore(), APSConformityScore(), RAPSConformityScore(),
    NaiveConformityScore(), TopKConformityScore()
]
wrong_cs_list = [object(), "LAC", 1]
valid_method_list = ['naive', 'aps', 'raps', 'lac', 'top_k']
all_method_list = valid_method_list + [None]
wrong_method_list = ['naive_', 'aps_', 'raps_', 'lac_', 'top_k_']

REGULARIZATION_PARAMETERS = [
    [.001, [1]],
    [[.01, .2], [1, 3]],
    [.1, [2, 4]]
]

X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 1, 2, 1, 2])
y_toy_string = np.array(["0", "0", "1", "0", "1", "1", "2", "1", "2"])

n_classes = 4
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_classes=n_classes,
    random_state=random_state,
)


def test_error_mother_class_initialization() -> None:
    """
    Test that the mother class BaseClassificationScore cannot be instantiated.
    """
    with pytest.raises(TypeError):
        BaseClassificationScore()  # type: ignore


@pytest.mark.parametrize("conformity_score", cs_list)
def test_check_classification_conformity_score(
    conformity_score: Optional[BaseClassificationScore]
) -> None:
    """
    Test that the function check_classification_conformity_score returns
    an instance of BaseClassificationScore when using conformity_score.
    """
    assert isinstance(
        check_classification_conformity_score(conformity_score),
        BaseClassificationScore
    )


@pytest.mark.parametrize("method", all_method_list)
def test_check_classification_method(
    method: Optional[str]
) -> None:
    """
    Test that the function check_classification_conformity_score returns
    an instance of BaseClassificationScore when using method.
    """
    assert isinstance(
        check_classification_conformity_score(method=method),
        BaseClassificationScore
    )


@pytest.mark.parametrize("method", valid_method_list)
@pytest.mark.parametrize("conformity_score", cs_list)
def test_check_conflict_parameters(
    method: Optional[str],
    conformity_score: Optional[BaseClassificationScore]
) -> None:
    """
    Test that the function check_classification_conformity_score raises
    a warning when both method and conformity_score are provided.
    """
    if conformity_score is None:
        return
    with pytest.warns(
        UserWarning,
        match="WARNING: the `conformity_score` parameter takes precedence*"
    ):
        check_classification_conformity_score(
            method=method, conformity_score=conformity_score
        )


@pytest.mark.parametrize("method", wrong_method_list)
def test_check_wrong_classification_method(
    method: Optional[str]
) -> None:
    """
    Test that the function check_classification_conformity_score raises
    a ValueError when using a wrong method.
    """
    with pytest.raises(
        ValueError, match="(Invalid method.)|(Invalid conformity score.)*"
    ):
        check_classification_conformity_score(method=method)


@pytest.mark.parametrize("score", wrong_cs_list)
def test_check_wrong_classification_score(
    score: Any
) -> None:
    """
    Test that the function check_classification_conformity_score raises
    a ValueError when using a wrong score.
    """
    with pytest.raises(ValueError, match="Invalid conformity_score argument*"):
        check_classification_conformity_score(conformity_score=score)


@pytest.mark.parametrize("cv", ['prefit', 'split'])
@pytest.mark.parametrize("size_raps", [0.2, 0.5, 0.8])
def test_check_depreciated_size_raps(size_raps: float, cv: str) -> None:
    """
    Test that the function check_classification_conformity_score raises
    a DeprecationWarning when using size_raps.
    """
    clf = LogisticRegression().fit(X, y)
    mapie_clf = MapieClassifier(
        estimator=clf, conformity_score=RAPSConformityScore(), cv=cv
    )
    with pytest.warns(
        DeprecationWarning,
        match="The parameter `size_raps` is deprecated.*"
    ):
        mapie_clf.fit(X, y, size_raps=size_raps)


@pytest.mark.parametrize("k_lambda", REGULARIZATION_PARAMETERS)
def test_regularize_conf_scores_shape(k_lambda) -> None:
    """
    Test that the conformity scores have the correct shape.
    """
    lambda_, k = k_lambda[0], k_lambda[1]
    conf_scores = np.random.rand(100, 1)
    cutoff = np.cumsum(np.ones(conf_scores.shape)) - 1
    reg_conf_scores = RAPSConformityScore._regularize_conformity_score(
        k, lambda_, conf_scores, cutoff
    )

    assert reg_conf_scores.shape == (100, 1, len(k))


def test_get_true_label_cumsum_proba_shape() -> None:
    """
    Test that the true label cumsumed probabilities
    have the correct shape.
    """
    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)
    mapie_clf = MapieClassifier(
        estimator=clf, random_state=random_state
    )
    mapie_clf.fit(X, y)
    classes = mapie_clf.classes_
    cumsum_proba, cutoff = APSConformityScore.get_true_label_cumsum_proba(
        y, y_pred, classes
    )
    assert cumsum_proba.shape == (len(X), 1)
    assert cutoff.shape == (len(X), )


def test_get_true_label_cumsum_proba_result() -> None:
    """
    Test that the true label cumsumed probabilities
    are the expected ones.
    """
    clf = LogisticRegression()
    clf.fit(X_toy, y_toy)
    y_pred = clf.predict_proba(X_toy)
    mapie_clf = MapieClassifier(
        estimator=clf, random_state=random_state
    )
    mapie_clf.fit(X_toy, y_toy)
    classes = mapie_clf.classes_
    cumsum_proba, cutoff = APSConformityScore.get_true_label_cumsum_proba(
        y_toy, y_pred, classes
    )
    np.testing.assert_allclose(
        cumsum_proba,
        np.array(
            [
                y_pred[0, 0], y_pred[1, 0],
                y_pred[2, 0] + y_pred[2, 1],
                y_pred[3, 0] + y_pred[3, 1],
                y_pred[4, 1], y_pred[5, 1],
                y_pred[6, 1] + y_pred[6, 2],
                y_pred[7, 1] + y_pred[7, 2],
                y_pred[8, 2]
            ]
        )[:, np.newaxis]
    )
    np.testing.assert_allclose(cutoff, np.array([1, 1, 2, 2, 1, 1, 2, 2, 1]))


@pytest.mark.parametrize("k_lambda", REGULARIZATION_PARAMETERS)
@pytest.mark.parametrize("include_last_label", [True, False])
def test_get_last_included_proba_shape(k_lambda, include_last_label):
    """
    Test that the outputs of _get_last_included_proba method
    have the correct shape.
    """
    lambda_, k = k_lambda[0], k_lambda[1]
    if len(k) == 1:
        thresholds = .2
    else:
        thresholds = np.random.rand(len(k))
    thresholds = cast(NDArray, check_alpha(thresholds))
    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred_proba = clf.predict_proba(X)
    y_pred_proba = np.repeat(
        y_pred_proba[:, :, np.newaxis], len(thresholds), axis=2
    )

    y_p_p_c, y_p_i_l, y_p_p_i_l = \
        RAPSConformityScore._get_last_included_proba(
            RAPSConformityScore(), y_pred_proba, thresholds,
            include_last_label, lambda_=lambda_, k_star=k
        )

    assert y_p_p_c.shape == (len(X), len(np.unique(y)), len(thresholds))
    assert y_p_i_l.shape == (len(X), 1, len(thresholds))
    assert y_p_p_i_l.shape == (len(X), 1, len(thresholds))
