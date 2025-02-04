from typing import Callable, Dict, Any, Optional, Tuple
from mapie._typing import NDArray
import inspect
from sklearn.model_selection import ShuffleSplit


def train_test_split_shuffle(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:

    splitter = ShuffleSplit(n_splits=1,
                            test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(splitter.split(X))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def filter_params(
    function: Callable,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if params is None:
        return {}

    model_params = inspect.signature(function).parameters
    return {k: v for k, v in params.items() if k in model_params}
