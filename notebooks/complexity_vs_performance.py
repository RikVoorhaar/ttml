# %%
"""Script for making plots comparing performance on test set, inference speed,
and model complexity,"""

"""We have already shown the correlation between model complexity and inference
speed. This doesn't include other popular models trained on the same dataset
such as random forests, XGBoost or MLP. We should make a plot that compares for
different models:
- number of parameters vs. inference speed
- number of parameters vs. loss

We can do this on airfoil, since we get good performance there, and for the TTML
we already have good data there. (At least for number of parameters vs. loss,
number of parameters vs. inference speed can be done without training any
model).

For TTML we will always take the best result among each type of init with a fixed
number rank and number of thresholds. We should however redo the inference speed
experiments in a more controlled setting.

For XGBoost, MLP and RF, we should vary the number of parameters in a grid
search. We can control the same hyperparameters as we did in the
hyperopt_benchmark script, although now we also want to control the MLP.
"""

"""First step, let's load in the best results for each TTML, store it in a new
CSV, but redo the inference speed tests using just random TTML's and better
averaging procedure."""

import os.path
import sys

sys.path.insert(1, "..")

import numpy as np
import pandas as pd
from datasets import load_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ttml.ttml import TTML
import matplotlib.pyplot as plt

from benchmarks import test_speed

BENCHMARK_CSV = "results/benchmark_complexity.csv"
DATASET_FOLDER = "../datasets/data"
dataset = load_data.dataset_loaders["airfoil"](DATASET_FOLDER)
X = dataset["X"]
y = dataset["y"]

# Redo speed experiments on `benchmark_all.csv`. This is necessary because the
# inference speed metric is garbage when running on the cluster. We also only
# keep the best result in terms of error for each value of `estimator_kwargs`.
if not os.path.isfile(BENCHMARK_CSV):
    old_results = pd.read_csv("results/benchmark_all.csv")
    old_results = old_results[old_results["dataset_name"] == "airfoil"]
    old_results = old_results[
        old_results["estimator_name"].str.startswith("ttml")
    ]
    gb = old_results.groupby("estimator_kwargs")
    old_results = old_results.loc[gb["error_mean"].idxmin()]

    speeds = pd.Series(index=old_results.estimator_kwargs, dtype=float)
    num_params = pd.Series(index=old_results.estimator_kwargs, dtype=float)
    for kwargs in tqdm(list(speeds.index)):
        tt_rank, num_thresh = kwargs.split("-")
        tt_rank = int(tt_rank)
        num_thresh = int(num_thresh)

        speeds_run = []
        num_params_run = []
        for i in range(12):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=179 + i
            )
            X_train2, X_val, y_train2, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=179 + i
            )
            ttml = TTML.random_from_data(X_train2, tt_rank, num_thresh)
            speed = test_speed(ttml, X)
            num_params_run.append(ttml.num_params)
            speeds_run.append(speed)

        speeds.loc[kwargs] = np.mean(speeds_run)
        num_params.loc[kwargs] = np.mean(num_params_run)

    speeds.name = "speed"
    num_params.name = "num_params"
    new_results = old_results.copy().drop("speed", axis=1)
    new_results = new_results.merge(speeds, on="estimator_kwargs")
    new_results = new_results.copy().drop("num_params", axis=1)
    new_results = new_results.merge(num_params, on="estimator_kwargs")
    new_results = new_results[old_results.columns]
    new_results["estimator_name"] = "ttml"
    new_results.to_csv(BENCHMARK_CSV, index=False)
else:
    new_results = pd.read_csv(BENCHMARK_CSV)
# %%
"""Now we need to run new experiments for all those other guys.
We can use `best_args` to check which experiments have already been run."""
import benchmarks
from itertools import product
from datetime import datetime

benchmarks.BENCHMARK_CSV = BENCHMARK_CSV
hyper_param_space = {
    "xgb": {
        "eta": np.logspace(-4, -0.3, 10),
        "max_depth": [6, 8, 12, 14, 16],
        "n_estimators": [2 ** i for i in range(13)],
    },
    "rf": {
        "max_leaf_nodes": [10, 20, 50, 100, 200, 400, 800, None],
        "max_depth": [5, 8, 10, 14, 20, 25, 40, 60, 80, 100, None],
        "n_estimators": [10, 50, 100, 250, 500, 1000, 2000],
    },
    "mlp": {
        "lr": np.logspace(-6, -3, 10),
        "layer_size": [20, 50, 75, 120, 200, 300, 500],
        "num_hidden_layers": [1, 2, 3],
    },
}
jobs = []
jobs_skipped = 0
for estimator_name, hyperparams in hyper_param_space.items():
    already_done_kwargs = set()
    estim_results = new_results[new_results["estimator_name"] == estimator_name]
    for x in estim_results["best_args"]:
        already_done_kwargs.add(tuple(eval(x).values()))

    param_vals = product(*hyperparams.values())
    for row in param_vals:
        # job = {'estimator_name':estimator_name}
        kwargs = {}
        for key, val in zip(hyperparams.keys(), row):
            kwargs[key] = val
        if tuple(eval(str(kwargs)).values()) not in already_done_kwargs:
            jobs.append((estimator_name, kwargs))
        else:
            jobs_skipped += 1
print(
    f"Doing {len(jobs)} jobs. Skipping {jobs_skipped} previously performed jobs"
)
# %%
for i, job in enumerate(jobs):
    print("@" * 80)
    print(f"{datetime.now()}  job {i}/{len(jobs)}: ")
    print(f"  {job}")
    print("@" * 80)
    estimator_name, kwargs = job
    benchmarks.do_benchmark(
        "airfoil", estimator_name, estimator_hyperparams=kwargs, force=True
    )
# %%
new_results[new_results["estimator_name"] == "ttml"].sort_values("error_mean")

old_results = pd.read_csv("results/benchmark_all.csv")
old_results = old_results[old_results["dataset_name"] == "airfoil"]
old_results[old_results["estimator_name"].str.startswith("ttml")].sort_values(
    "error_mean"
)


# %%
"""Plot results, this should go into new notebook to be honest"""
MIN_ERROR = 5

plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
ls_dict = {"ttml": "solid", "xgb": "dashdot", "rf": "dashed", "mlp": "dotted"}
for estimator_name in new_results["estimator_name"].unique():
    results = new_results[new_results["estimator_name"] == estimator_name]

    # plt.plot(results['num_params'],results['error_mean'],'.')
    errors = []
    best_error = MIN_ERROR
    n_params = []
    results.sort_values("num_params", inplace=True)
    for row in results[["error_mean", "num_params"]].itertuples(False):
        error, n_param = row
        if error < best_error:
            best_error = error
            errors.append(error)
            n_params.append(n_param)
    plt.plot(n_params, errors, ls=ls_dict[estimator_name], label=estimator_name)
plt.xlabel("Number of parameters")
plt.ylabel("Test error (MSE)")
plt.xscale("log")
plt.xlim(100, None)
plt.legend()

plt.subplot(1, 3, 2)
for estimator_name in new_results["estimator_name"].unique():
    results = new_results[new_results["estimator_name"] == estimator_name]

    # plt.plot(results['num_params'],results['error_mean'],'.')
    speeds = []
    best_speed = -np.inf
    n_params = []
    results.sort_values("num_params", inplace=True, ascending=False)
    for row in results[["error_mean", "speed", "num_params"]].itertuples(False):
        error, speed, n_param = row
        if error < MIN_ERROR and speed > best_speed:
            best_speed = speed
            speeds.append(speed)
            n_params.append(n_param)
    plt.plot(n_params, speeds, ls=ls_dict[estimator_name], label=estimator_name)
plt.xlabel("Number of parameters")
plt.ylabel("Inference speed (samples/s)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.subplot(1, 3, 3)
for estimator_name in new_results["estimator_name"].unique():
    results = new_results[new_results["estimator_name"] == estimator_name]

    # plt.plot(results['num_params'],results['error_mean'],'.')
    speeds = []
    best_speed = -np.inf
    errors = []
    results.sort_values("error_mean", inplace=True, ascending=True)
    for row in results[["error_mean", "speed", "num_params"]].itertuples(False):
        error, speed, n_param = row
        if error < MIN_ERROR and speed > best_speed:
            best_speed = speed
            speeds.append(speed)
            errors.append(error)
    plt.plot(errors, speeds, ls=ls_dict[estimator_name], label=estimator_name)
plt.xlabel("Test error (MSE)")
plt.ylabel("Inference speed (samples/s)")
plt.yscale("log")
# plt.ylim(2, 8)
# plt.xlim(100, None)
plt.legend()

plt.savefig(
    f"./figures/complexity_comparison.pdf",
    format="pdf",
    bbox_inches="tight",
)


# %%
rank_fun = lambda s: int(s.split("-")[0])
thresh_fun = lambda s: int(s.split("-")[1])
df_ttml = new_results[new_results["estimator_name"] == "ttml"].copy()

df_ttml["rank"] = df_ttml.estimator_kwargs.map(rank_fun)
df_ttml["num_thresh"] = df_ttml.estimator_kwargs.map(thresh_fun)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
ranks = df_ttml["rank"].unique()
for rank in ranks:
    if rank <= 9:
        marker = "+"
    else:
        marker = "x"
    df_rank = df_ttml[df_ttml["rank"] == rank]
    plt.plot(
        df_rank["num_params"], df_rank["speed"], marker=marker, linestyle="None"
    )
plt.title("Inference speed vs. model complexity")
plt.xscale("log")
plt.xlabel("Number of parameters")
plt.ylabel("Samples / s")

plt.subplot(1, 2, 2)
import scipy.stats

exponent = 2
res = scipy.stats.linregress(df_ttml["rank"] ** (1 / exponent), df_ttml["speed"])
X_plot = np.unique(df_ttml["rank"] ** (1 / exponent))
plt.plot(
    X_plot ** exponent, res.slope * X_plot + res.intercept, c="k", alpha=0.5
)
print(res)

for rank in ranks:
    if rank <= 11:
        marker = "+"
    else:
        marker = "x"
    df_rank = df_ttml[df_ttml["rank"] == rank]
    plt.plot(df_rank["rank"], df_rank["speed"], marker=marker, linestyle="None")
# plt.plot(df_ttml["rank"], df_ttml["speed"], ".")
plt.title("Inference speed vs. model complexity")
plt.xlabel("TT-rank")
plt.ylabel("Samples / s")
plt.xticks(np.arange(2, 20, 1))

plt.savefig(
    f"./figures/inference_speed_old_style.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)

rank_fun = lambda s: int(s.split("-")[0])
thresh_fun = lambda s: int(s.split("-")[1])
df_ttml = new_results[new_results["estimator_name"] == "ttml"].copy()

df_ttml["rank"] = df_ttml.estimator_kwargs.map(rank_fun)
df_ttml["num_thresh"] = df_ttml.estimator_kwargs.map(thresh_fun)

ranks = df_ttml["rank"].unique()
num_thresh_vals = df_ttml["num_thresh"].unique()
df_ttml = df_ttml.sort_values(by=["rank","num_thresh"])
for num_thresh in num_thresh_vals:
    df_num_thresh = df_ttml[df_ttml["num_thresh"] == num_thresh].copy()
    df_num_thresh["speed"] = df_num_thresh["speed"]/1e6
    plt.plot(df_num_thresh["num_params"], df_num_thresh["speed"],".-",label=f"{num_thresh} thresholds")
# for rank in ranks:
#     if rank <= 9:
#         marker = "+"
#     else:
#         marker = "x"
#     df_rank = df_ttml[df_ttml["rank"] == rank]
#     plt.plot(
#         df_rank["num_params"], df_rank["speed"], marker=marker, linestyle="None"
#     )
plt.suptitle("\nInference speed vs. model complexity")
plt.xscale("log")
plt.xlabel("Number of parameters")
plt.ylabel(r"Samples $\times$ 1 million / s")

plt.subplot(1, 2, 2)
# plt.title("Inference speed vs. model complexity")
df_ttml = df_ttml.sort_values(by=["rank","num_thresh"])
num_thresh_vals = df_ttml["num_thresh"].unique()
ranks = df_ttml["rank"].unique()
for num_thresh in num_thresh_vals:
    df_num_thresh = df_ttml[df_ttml["num_thresh"] == num_thresh].copy()
    df_num_thresh["speed"] = df_num_thresh["speed"]/1e6
    plt.plot(ranks, df_num_thresh["speed"],".-",label=f"{num_thresh} thresholds")

plt.legend()
plt.ylabel(r"Samples $\times$ 1 million / s")
plt.xlabel("TT-rank")
plt.xticks(np.arange(2, 20, 1));

plt.savefig(
    f"./figures/inference_speed.pdf",
    format="pdf",
    bbox_inches="tight",
)
# %%
