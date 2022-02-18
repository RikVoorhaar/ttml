# %%
"""This scripts fits a ttml model to every permutation of features for a given
datset. Does this multiple time to obtain statistics."""

import sys
import os
from tqdm import tqdm
import itertools
from itertools import product
import numpy as np
import pandas as pd
import csv


sys.path.insert(1, "..")
from datasets import load_data


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss
from ttml.ttml import TTML
from ttml.forest_compression import compress_forest_thresholds
from ttml.utils import univariate_kmeans
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)
import scipy.special
import matplotlib.pyplot as plt

NUM_BATCHES = 30

dataset_name = "airfoil"
DATASET_FOLDER = "../datasets/data"
BENCHMARK_CSV = f"permutation_{dataset_name}.csv"
dataset = load_data.dataset_loaders[dataset_name](DATASET_FOLDER)
X = dataset["X"]
y = dataset["y"]


def preprocessing(permutation, seed):
    X_permuted = X[:, permutation]
    X_train, X_test, y_train, y_test = train_test_split(
        X_permuted.astype(float),
        y.astype(float),
        test_size=0.15,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=seed,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    max_leaf_nodes = None
    max_depth = None
    if dataset["regression"]:
        forest_estim = RandomForestRegressor
        task = "regression"
        metric = mean_squared_error
    else:
        forest_estim = RandomForestClassifier
        task = "classification"
        metric = log_loss
    forest = forest_estim(
        n_estimators=256,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
    )
    forest.fit(X_train, y_train)
    return forest, X_train, X_val, X_test, y_train, y_val, y_test


def test(permutation):
    # Check if benchmark has been done, skip otherwise
    if os.path.isfile(BENCHMARK_CSV):
        df = pd.read_csv(BENCHMARK_CSV)
        selector = df["permutation"] == str(permutation)
        if sum(selector) > 0:  # benchmark has been done before
            return

    errors = []

    for i in range(NUM_BATCHES):
        forest, X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(
            permutation, 179 + i
        )
        thresholds = TTML.thresholds_from_data(X_train, 40)

        ttml = TTML.fit(
            X_train,
            y_train,
            forest,
            X_val = X_val,
            y_val = y_val,
            max_rank=6,
            task="regression",
            _thresholds=thresholds,
        )

        y_pred = ttml.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        errors.append(error)
    return errors


permutations = list(itertools.permutations(range(X.shape[1])))
for permutation in tqdm(permutations):
    errors = test(permutation)
    if errors is None:
        continue
    error_mean = np.mean(errors)
    error_min = np.min(errors)
    error_std = np.std(errors)

    with open(BENCHMARK_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
        if csvfile.tell() == 0:
            writer.writerow(
                [
                    "permutation",
                    "error_mean",
                    "error_best",
                    "error_std",
                    "individual_trials",
                ]
            )
        writer.writerow([permutation, error_mean, error_min, error_std, errors])

# %%
