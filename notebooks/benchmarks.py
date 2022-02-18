"""Script for benchmarking and comparing the TTML estimators to baseline
estimators. """
import argparse
import csv
import datetime
import itertools
from joblib import parallel_backend
import os.path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import data
import xgboost as xgb
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
)

from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(1, "..")

from datasets import load_data
from ttml.ttml import TTMLClassifier, TTMLRegressor

DATASET_FOLDER = "../datasets/data"
BENCHMARK_CSV = "results/benchmark_all.csv"
THRESHOLD_METHOD = "data"
NUM_TRIALS = 12
USE_TQDM = False

estimator_dict = {
    "classification": {
        "ttml": TTMLClassifier,
        "xgb": xgb.XGBClassifier,
        "rf": RandomForestClassifier,
        "mlp": MLPClassifier,
        "gp": GaussianProcessClassifier,
        "metric": log_loss,
    },
    "regression": {
        "ttml": TTMLRegressor,
        "xgb": xgb.XGBRegressor,
        "rf": RandomForestRegressor,
        "mlp": MLPRegressor,
        "gp": GaussianProcessRegressor,
        "metric": mean_squared_error,
    },
}

dt = datetime.datetime.now


def do_benchmark(
    dataset_name,
    estimator_name,
    estimator_kwargs=None,
    estimator_hyperparams=None,
    seed=179,
    force=False,
):
    dataset = load_data.dataset_loaders[dataset_name](DATASET_FOLDER)
    X = dataset["X"].astype(float)
    y = dataset["y"].astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if dataset["regression"]:
        task = "regression"
    else:
        task = "classification"

    # Check if benchmark has been done to avoid double work
    if not force and os.path.isfile(BENCHMARK_CSV):
        df = pd.read_csv(BENCHMARK_CSV)
        selector = (df["dataset_name"] == dataset_name) & (
            df["estimator_name"] == estimator_name
        )
        if estimator_kwargs is not None:
            selector = selector & (df["estimator_kwargs"] == estimator_kwargs)
        selector = selector & (df["threshold_method"] == THRESHOLD_METHOD)
        if sum(selector) > 0:  # benchmark has been done before
            print("Benchmark has been done before, skipping...")
            return

    with parallel_backend("threading", -1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings...
            output = benchmark_general(
                X,
                y,
                task,
                estimator_name,
                seed,
                estimator_kwargs=estimator_kwargs,
                estimator_hyperparams=estimator_hyperparams,
            )

    with open(BENCHMARK_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
        if csvfile.tell() == 0:
            writer.writerow(
                [
                    "dataset_name",
                    "estimator_name",
                    "estimator_kwargs",
                    "error_mean",
                    "error_std",
                    "error_best",
                    "num_params",
                    "speed",
                    "best_args",
                    "individual_trials",
                    "threshold_method",
                ]
            )
        writer.writerow(
            [dataset_name, estimator_name, estimator_kwargs]
            + list(output)
            + [THRESHOLD_METHOD]
        )
    return output


def make_gp(X, y, task, kwargs):
    kernel_dict = {
        "RQ": RationalQuadratic(),
        "RBF": RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
        "Matern-1.5": Matern(nu=1.5),
        "Matern-2.0": Matern(nu=2.0),
        "Matern-2.5": Matern(nu=2.5),
    }
    gp_estimator = estimator_dict[task]["gp"](
        kernel=kernel_dict[kwargs["kernel"]]
    )
    X_train = X
    y_train = y
    if (
        task == "classification" and len(X) > 1000
    ):  # fitting is O(n^3), so subsample for speed
        inds = np.random.choice(len(X), size=1000, replace=False)
        X_train = X[inds]
        y_train = y[inds]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore convergence warnings
        gp_estimator.fit(X_train, y_train)
    return gp_estimator


def make_mlp(X, y, task, kwargs):
    mlp_estimator = estimator_dict[task]["mlp"](
        hidden_layer_sizes=[kwargs["layer_size"]] * kwargs["num_hidden_layers"],
        learning_rate_init=kwargs["lr"],
        max_iter=1000,
        early_stopping=True,
        solver="lbfgs",
    )
    mlp_estimator.fit(X, y)
    return mlp_estimator


def make_rf(X, y, task, kwargs):
    try:
        n_estimators = kwargs["n_estimators"]
    except KeyError:
        n_estimators = 256
    forest = estimator_dict[task]["rf"](
        n_estimators=n_estimators,
        max_leaf_nodes=kwargs["max_leaf_nodes"],
        max_depth=kwargs["max_depth"],
    )
    forest.fit(X, y)
    return forest


def make_xgb(X, y, task, kwargs):
    if task == "regression":
        eval_metric = None
    else:
        eval_metric = "logloss"
    xgb_estimator = estimator_dict[task]["xgb"](
        learning_rate=kwargs["eta"],
        n_estimators=kwargs["n_estimators"],
        max_depth=kwargs["max_depth"],
        eval_metric=eval_metric,
        use_label_encoder=False,
    )
    xgb_estimator.fit(X, y)
    return xgb_estimator


def fit_ttml(X, y, task, kwargs):
    base_estim = base_estimator_routines[kwargs["estimator_name"]](
        X, y, task, kwargs
    )
    ttml = estimator_dict[task]["ttml"](
        base_estim,
        max_rank=kwargs["max_rank"],
        num_thresholds=kwargs["num_thresholds"],
    )
    ttml.fit(X, y)
    return ttml


base_estimator_routines = {
    "ttml_gp": make_gp,
    "ttml_mlp1": make_mlp,
    "ttml_mlp2": make_mlp,
    "ttml_rf": make_rf,
    "ttml_xgb": make_xgb,
}

fit_routines = {"ttml": fit_ttml, "xgb": make_xgb, "rf": make_rf, "mlp": make_mlp}

ttml_gp_params = {
    "kernel": ["RQ", "RBF", "Matern-1.5", "Matern-2.0", "Matern-2.5"]
}
ttml_mlp1_params = {
    "lr": [1e-3, 1e-4, 1e-5],
    "layer_size": [75, 120, 200, 300],
    "num_hidden_layers": [1],
}
ttml_mlp2_params = {
    "lr": [1e-3, 1e-4, 1e-5],
    "layer_size": [75, 120, 200, 300],
    "num_hidden_layers": [2],
}
ttml_rf_params = {
    "max_leaf_nodes": [200, 400, 800, None],
    "max_depth": [20, 40, 60, None],
}
ttml_xgb_params = {
    "eta": [0.02, 0.05, 0.1, 0.2],
    "max_depth": [6, 8, 12, 14, 16],
    "n_estimators": [200],
}
rf_params = {
    "max_leaf_nodes": [50, 100, 200, 400, 800, None],
    "max_depth": [10, 20, 40, 60, 80, None],
}
mlp_params = {
    "lr": np.logspace(-6, -3, 5),
    "layer_size": [50, 120, 200, 300],
    "num_hidden_layers": [1, 2, 3],
}
xgb_params = {
    "eta": np.logspace(-4, -0.3, 20),
    "max_depth": [6, 8, 12, 14, 16],
}
estimator_params = {
    "ttml_gp": ttml_gp_params,
    "ttml_mlp1": ttml_mlp1_params,
    "ttml_mlp2": ttml_mlp2_params,
    "ttml_rf": ttml_rf_params,
    "ttml_xgb": ttml_xgb_params,
    "xgb": xgb_params,
    "rf": rf_params,
    "mlp": mlp_params,
}


def benchmark_general(
    X,
    y,
    task,
    estimator_name,
    seed,
    estimator_kwargs=None,
    estimator_hyperparams=None,
):
    try:
        if estimator_kwargs is not None:
            estimator_kwargs = tuple(
                int(a) for a in estimator_kwargs.split("-")
            )
        if estimator_name.startswith("ttml"):
            if estimator_hyperparams is None:
                assert len(estimator_kwargs) >= 2
                max_rank = estimator_kwargs[0]
                num_thresholds = estimator_kwargs[1]
            fit_routine = fit_routines["ttml"]
        elif estimator_name == "xgb":
            if estimator_hyperparams is None:
                assert len(estimator_kwargs) >= 1
                n_estimators = estimator_kwargs[0]
            fit_routine = fit_routines["xgb"]
        elif estimator_name == "rf":
            fit_routine = fit_routines["rf"]
        elif estimator_name == "mlp":
            fit_routine = fit_routines["mlp"]
        else:
            raise ValueError(f"Unknown model {estimator_name}")
    except AssertionError:
        raise ValueError("arguments are in invalid format.")

    # Initial split for hyper parameter optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed
    )
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=seed
    )

    if estimator_hyperparams is None:
        print(f"{dt()}  Start hyperparameter optimization")
        param_list = list(
            itertools.product(*estimator_params[estimator_name].values())
        )
        if USE_TQDM:
            param_list = tqdm(param_list)
        best_error = np.inf
        best_kwargs = None

        for i, params in enumerate(param_list):
            kwargs = {
                key: param
                for key, param in zip(
                    estimator_params[estimator_name].keys(), params
                )
            }
            kwargs["estimator_name"] = estimator_name
            if estimator_name.startswith("ttml"):
                kwargs["max_rank"] = max_rank
                kwargs["num_thresholds"] = num_thresholds
            elif estimator_name == "xgb":
                kwargs["n_estimators"] = n_estimators

            try:
                model = fit_routine(X_train2, y_train2, task, kwargs)
            except ValueError:
                print(
                    "ERROR: Got an error for the following params, skiping..."
                )
                print(kwargs)
                continue

            if task == "regression":
                error = mean_squared_error(y_val, model.predict(X_val))
            else:
                error = log_loss(y_val, model.predict_proba(X_val))
            if error < best_error:
                best_error = error
                best_kwargs = kwargs
            if not USE_TQDM:
                print(f"{dt()}  {i+1}/{len(param_list)}")
    else:
        print(f"{dt()}  Using kwargs {estimator_hyperparams}")
        best_kwargs = estimator_hyperparams

    print(f"\n{dt()}  Testing accuracy")
    errors = []
    trials_it = range(NUM_TRIALS)
    if USE_TQDM:
        trials_it = tqdm(trials_it)
    for i in trials_it:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=seed + i
        )
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=seed + i
        )
        model = fit_routine(X_train2, y_train2, task, best_kwargs)

        if task == "regression":
            error = mean_squared_error(y_test, model.predict(X_test))
        else:
            error = log_loss(y_test, model.predict_proba(X_test))
        errors.append(error)
        if not USE_TQDM:
            print(f"{dt()}  {i+1}/{NUM_TRIALS}")

    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_best = np.min(errors)
    print(f"\n{error_mean=:.5e}, {error_std=:.3e}, {error_best=:.5e}")

    if estimator_name.startswith("ttml"):
        # num_params = model.ttml_.tt.num_params()
        num_params = model.ttml_.num_params
    elif estimator_name == "xgb":
        num_params = model.get_booster().trees_to_dataframe().size
    elif estimator_name == "rf":
        num_params = sum(
            [
                len(tree.tree_.__getstate__()["nodes"]) * 8
                for tree in model.estimators_
            ]
        )
    elif estimator_name == "mlp":
        num_params = np.sum(
            [np.prod(c.shape) for c in model.coefs_]
            + [np.prod(c.shape) for c in model.intercepts_]
        )

    print(f"Number of parameters: {num_params:.3e}")

    print(f"\n{dt()} Testing inference speed")
    speed = test_speed(model, X)
    print(f"\n inference speed: {speed:.5e} samples / s")

    return (
        error_mean,
        error_std,
        error_best,
        num_params,
        speed,
        best_kwargs,
        errors,
    )


def test_speed(estimator, X):
    from time import perf_counter_ns

    num_samples = len(X)
    num_its = int(10 ** 6 / num_samples)
    tot_samples = num_samples * num_its
    X_big = np.concatenate([X]*num_its)

    t0 = perf_counter_ns()
    estimator.predict(X_big)
    t1 = perf_counter_ns()
    time = (t1 - t0) * 1e-9
    speed = tot_samples / time
    return speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark an estimator on a chosen dataset"
    )
    parser.add_argument(
        "dataset_name",
        metavar="name",
        nargs="?",
        type=str,
        help="name of dataset, e.g. 'airfoil'",
    )
    parser.add_argument(
        "estimator_name",
        nargs="?",
        type=str,
        help=f"estimator name, one of {list(estimator_params.keys())}",
    )
    parser.add_argument(
        "-a",
        "--args",
        help="comma separated list of args for TTML estimator. Kwargs are\
            num_thresholds, max_rank",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Don't check if benchmark has been done before.",
    )
    parser.add_argument(
        "-l",
        "--list",
        dest="list",
        action="store_true",
        help="list available datasets and quit",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        nargs="?",
        type=int,
        default=179,
        help="Seed for train/validation split",
    )

    args = parser.parse_args()
    if args.list:
        print("Available datasets:")
        print(list(load_data.dataset_loaders.keys()))
    else:
        if (args.dataset_name is None) or (args.estimator_name is None):
            print("Invalid syntax.")
            parser.print_help()
        else:
            do_benchmark(
                args.dataset_name,
                args.estimator_name,
                estimator_kwargs=args.args,
                seed=args.seed,
                force=args.force,
            )
