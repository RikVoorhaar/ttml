# %%

import sys
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd


sys.path.insert(1, "..")
from datasets import load_data


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, log_loss
from ttml.ttml import TTML
from ttml.forest_compression import compress_forest_thresholds
from ttml.utils import univariate_kmeans, thresholds_from_data
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

X_train, X_test, y_train, y_test = train_test_split(
    X.astype(float),
    y.astype(float),
    test_size=0.15,
    random_state=179,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=179,
)
del X, y  # delete X,y to avoid making stupid mistakes

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
# %%
def data_kmeans(num_thresh):
    return thresholds_from_data(
        X_train, num_thresh, min_samples=3, strategy="kmeans"
    )


def data_uniform(num_thresh):
    return thresholds_from_data(
        X_train, num_thresh, min_samples=3, strategy="uniform"
    )


def data_quantile_ttml(num_thresh):
    return TTML.thresholds_from_data(X_train, num_thresh)


def forest_kmeans(num_thresh):
    _, thresholds = compress_forest_thresholds(
        forest, num_thresh, use_univariate_kmeans=True
    )
    return thresholds


def forest_quantile(num_thresh):
    _, thresholds = compress_forest_thresholds(
        forest, num_thresh, use_univariate_kmeans=False
    )
    return thresholds


threshold_methods = {
    "Forest kmeans": forest_kmeans,
    "Forest quantile": forest_quantile,
    "Data kmeans": data_kmeans,
    "Data uniform": data_uniform,
    "Data quantile": data_quantile_ttml,
}

results_df = pd.DataFrame(
    columns=["method", "mean_error", "min_error", "num_thresh", "num_params"]
)

# %%
hyperparams = product(np.arange(5, 51, 5), threshold_methods.keys())
for i, (num_thresh, name) in enumerate(tqdm(list(hyperparams))):
    X = dataset["X"]
    y = dataset["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(float),
        y.astype(float),
        test_size=0.15,
        random_state=179 + i,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=179 + i,
    )

    thresholds = threshold_methods[name](int(num_thresh))
    errors = []
    num_params = []
    for _ in range(15):
        ttml = TTML.fit(
            X_train,
            y_train,
            forest,
            X_val=X_val,
            y_val=y_val,
            max_rank=10,
            task=task,
            _thresholds=thresholds,
        )
        y_pred = ttml.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        errors.append(error)
        num_params.append(ttml.tt.num_params())

    results_df.loc[i] = (
        name,
        np.median(errors),
        np.min(errors),
        num_thresh,
        np.mean(num_params),
    )

# %%
results_df.to_csv("results/threshold_method.csv")
# %%
plt.figure(figsize=(10, 6))
for name in threshold_methods.keys():
    if name == 'Data uniform':
        continue
    df_name = results_df[results_df["method"] == name].copy()
    df_name.sort_values("num_params", inplace=True)
    plt.plot(df_name["num_params"], df_name["mean_error"], 'o--', label=name,)
plt.xlabel("Number of parameters")
plt.ylabel("Error (MSE)")
plt.legend()

plt.savefig(
    f"./figures/feature_space_discretization.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.show()

# plt.figure(figsize=(10, 6))
# for name in threshold_methods.keys():
#     df_name = results_df[results_df["method"] == name]
#     plt.plot(df_name["num_thresh"], df_name["mean_error"], label=name)
# plt.xlabel("num_thresh")
# plt.ylabel("error (MSE)")
# plt.legend()
# plt.show()
# %%
