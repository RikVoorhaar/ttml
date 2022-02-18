import autoray as ar
from ttml.utils import SUPPORTED_BACKENDS, convert_backend
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from ttml.ttml import TTMLClassifier, TTMLRegressor, TTML
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from ttml.tt_opt import _classification_loss


def make_dataset(N, task, backend="numpy"):
    "make an artificial dataset for testing"
    X = np.random.normal(size=(N, 5))
    X = np.concatenate([X, np.random.randint(-4, 5, size=(N, 1))], axis=1)
    X = X.astype(np.float64)
    if task == "regression":
        y = np.mean(X, axis=1)
    elif task == "classification":
        y = np.sum(X, axis=1) > 0

    y = y.astype(np.float64)
    y = np.float64(y)
    X = np.float64(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    if task == "regression":
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val = scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

    X_train = convert_backend(X_train, backend)
    y_train = convert_backend(y_train, backend)
    X_val = convert_backend(X_val, backend)
    y_val = convert_backend(y_val, backend)

    return X_train, X_val, y_train, y_val


@pytest.mark.parametrize("task", ("regression", "classification"))
@pytest.mark.parametrize("max_rank", (2, 4))
@pytest.mark.parametrize("num_thresholds", (10, 20))
@pytest.mark.parametrize("estimator_type", ("rf", "xgb"))
def test_ttml_fit(task, max_rank, num_thresholds, estimator_type):
    X_train, X_val, y_train, y_val = make_dataset(1000, task, "numpy")

    if task == "regression":
        if estimator_type == "xgb":
            estimator = XGBRegressor()
        else:
            estimator = RandomForestRegressor()
        ttml = TTMLRegressor(
            estimator, max_rank=max_rank, num_thresholds=num_thresholds
        )
        ttml.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        error = mean_squared_error(ttml.predict(X_val), y_val)
        assert error < 1
    elif task == "classification":
        if estimator_type == "xgb":
            estimator = XGBClassifier()
        else:
            estimator = RandomForestClassifier()
        ttml = TTMLClassifier(
            estimator, max_rank=max_rank, num_thresholds=num_thresholds
        )
        ttml.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        ttml = ttml.ttml_
        error = _classification_loss(
            ttml.tt, y_val, ttml.observed_indices(X_val), None, True
        )
        assert error < 1


def test_from_tree():
    X, _, y, _ = make_dataset(1000, "regression")
    dec_tree = DecisionTreeRegressor(max_leaf_nodes=40)
    dec_tree.fit(X, y)
    ttml = TTML.from_tree(dec_tree)
    error = np.linalg.norm(ttml.predict(X) - dec_tree.predict(X))
    assert error < 1e-8


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_tree_arithmetic(backend):
    X, _, _, _ = make_dataset(1000, "regression")
    ttml1 = TTML.random_from_data(
        X, 2, 4, categorical_features=[5], backend=backend
    )
    ttml2 = TTML.random_from_data(
        X, 2, 4, categorical_features=[5], backend=backend
    )
    pred1 = ttml1.predict(X)
    pred2 = ttml2.predict(X)

    ttml3 = ttml1 * 2
    assert ar.do("linalg.norm", ttml3.predict(X) - pred1 * 2) < 1e-8
    ttml3 = ttml3 / 2
    assert ar.do("linalg.norm", ttml3.predict(X) - pred1) < 1e-8

    ttml4 = ttml1 + ttml2
    assert ar.do("linalg.norm", ttml4.predict(X) - (pred1 + pred2)) < 1e-8
