#%%
"""Demonstrate the effect of compressing the number of thresholds of a random
forest."""


import sys
from numpy.linalg import LinAlgError
from tqdm import tqdm
import os.path
import numpy as np
import xgboost as xgb


sys.path.insert(1, "..")
from datasets import load_data
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss
from ttml.ttml import TTML
from ttml.tt_rlinesearch import TTLS
from ttml.forest_compression import compress_forest_thresholds
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)
import scipy.special
import matplotlib.pyplot as plt

dataset_name = "airfoil"
DATASET_FOLDER = "../datasets/data"
dataset = load_data.dataset_loaders[dataset_name](DATASET_FOLDER)
X = dataset["X"]
y = dataset["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X.astype(float),
    y.astype(float),
    test_size=0.2,
    random_state=179,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

max_leaf_nodes = 100
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
    n_estimators=128,
    max_leaf_nodes=max_leaf_nodes,
    max_depth=max_depth,
)

forest.fit(X_train, y_train)


# %%
# %%
# %%_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss
from ttml.ttml import TTML
from ttml.tt_rlinesearch import TTLS
from ttml.forest_compression import compress_forest_thresholds
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)
import scipy.special
import matplotlib.pyplot as plt

dataset_name = "airfoil"
DATASET_FOLDER = "../datasets/data"
dataset = load_data.dataset_loaders[dataset_name](DATASET_FOLDER)
X = dataset["X"]
y = dataset["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X.astype(float),
    y.astype(float),
    test_size=0.2,
    random_state=179,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

max_leaf_nodes = 100
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
    n_estimators=128,
    max_leaf_nodes=max_leaf_nodes,
    max_depth=max_depth,
)

forest.fit(X_train, y_train)


def val_loss(forest):
    loss = metric(y_val, forest.predict(X_val))
    return loss


uncompresed_loss = val_loss(forest)
# %%
num_thresholds = np.arange(10, 100)
losses = []
for num_t in tqdm(num_thresholds):
    compressed_forest, _ = compress_forest_thresholds(forest, num_t)
    losses.append(val_loss(compressed_forest))
# %%
plt.figure(figsize=(10,6))
plt.plot(num_thresholds, losses, ".")
plt.axhline(uncompresed_loss, c="k", ls="--")
plt.xlabel("Number of decision boundaries")
plt.ylabel("Validation MSE")
# plt.savefig("./figures/forest_compression.pdf", format='pdf',bbox_inches='tight')
# %%
