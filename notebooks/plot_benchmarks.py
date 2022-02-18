# %%
"""Main script for making plot of benchmarks obtained by the `benchmarks.py` module"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, "..")

from datasets import load_data

DATASET_NAME = "shill_bidding"
DATASET_FOLDER = "../datasets/data"
BENCHMARK_CSV = "results/benchmark_all.csv"
df = pd.read_csv(BENCHMARK_CSV)
print(f"Available datasets: {df['dataset_name'].unique()}")
df = df[df["dataset_name"] == DATASET_NAME]
dataset = load_data.dataset_loaders[DATASET_NAME](DATASET_FOLDER)
if dataset["regression"]:
    task = "regression"
else:
    task = "classification"

rank_fun = lambda s: int(s.split("-")[0])
thresh_fun = lambda s: int(s.split("-")[1])


def apply_to_trials(df, func):
    results = []
    for trial in df["individual_trials"]:
        row = eval(trial)
        row = np.array(row)
        results.append(func(row))
    return pd.Series(np.array(results), df.index)


for threshold_method in ("data",):
    i = 1
    plt.figure(figsize=(17, 19))
    xgb_df = df.loc[df["estimator_name"] == "xgb"]
    xgb_best_median = np.min(apply_to_trials(xgb_df, np.median))
    try:
        rf_trials = df.loc[df["estimator_name"] == "rf"].iloc[0][
            "individual_trials"
        ]
        rf_median = np.median(eval(rf_trials))
    except IndexError:
        rf_median = 0

    best_medians = dict()

    for model, model_name in (
        ("ttml_xgb", "XGBoost"),
        ("ttml_rf", "random forest"),
        ("ttml_mlp1", "MLP (1 hidden layer)"),
        ("ttml_mlp2", "MLP (2 hidden layers)"),
        ("ttml_gp", "Gaussian Process"),
    ):
        df_rf = df[df.estimator_name == model].copy()
        df_rf = df_rf[df_rf["threshold_method"] == threshold_method]
        df_rf["rank"] = df_rf.estimator_kwargs.map(rank_fun)
        df_rf["num_thresh"] = df_rf.estimator_kwargs.map(thresh_fun)
        df_rf.sort_values("num_thresh", inplace=True)
        df_rf.sort_values("rank", inplace=True)

        if DATASET_NAME == "airfoil":
            num_thresholds = [15, 25, 50, 100, 150]
        elif DATASET_NAME == "concrete":
            num_thresholds = [15, 20, 25, 35, 50]
        elif DATASET_NAME == "shill_bidding":
            num_thresholds = [20, 25, 35, 50]
        else:
            num_thresholds = df_rf.num_thresh.unique()
        plt.subplot(3, 2, i)
        all_medians = []
        for t in num_thresholds:
            sub_df = df_rf[df_rf["num_thresh"] == t]
            medians = apply_to_trials(sub_df, np.median)
            all_medians.append(medians)

            plt.plot(sub_df["rank"], medians, label=f"#Thresholds={t}")
            best_medians[model_name] = (
                np.ones(np.max([len(t) for t in all_medians])) * np.inf
            )
            for t in all_medians:
                for j, m in enumerate(t):
                    if best_medians[model_name][j] > m:
                        best_medians[model_name][j] = m
        plt.axhline(xgb_best_median, c="k", ls="--", label="XGBoost")
        plt.axhline(rf_median, c="r", ls="--", label="Random Forest")
        plt.legend()
        plt.title(f"TT-rank vs. median error, initialized with {model_name}")
        plt.xticks(np.arange(2, 20, 1))
        plt.xlabel("TT-rank")
        if task == "regression":
            plt.ylabel("MSE")
        else:
            plt.ylabel("Cross entropy")

        i += 1
    plt.subplot(3, 2, i)
    plt.xticks(np.arange(2, 22, 1))
    plt.xlabel("TT-rank")
    if task == "regression":
        plt.ylabel("MSE")
    else:
        plt.ylabel("Cross entropy")
    if DATASET_NAME == "concrete":
        plt.ylim(15, 60)
    for model_name, medians in best_medians.items():
        plt.plot(np.arange(2, 2 + len(medians)), medians, label=model_name)
    plt.axhline(xgb_best_median, c="k", ls="--", label="XGBoost (baseline)")
    plt.axhline(rf_median, c="r", ls="--", label="Random Forest (baseline)")
    plt.title("TT-rank vs. best median error for each initialization model")
    plt.legend()

plt.savefig(
    f"./figures/{DATASET_NAME}_init_method.pdf",
    format="pdf",
    bbox_inches="tight",
)
# plt.show()
# %%
best_medians

# %%
plt.figure(figsize=(14, 10))
i = 1

# Plot rank vs. loss for different num_thresh
# plt.suptitle(f"Performance of estimator on '{DATASET_NAME}' dataset")
for model, model_name in (("ttml_xgb", "XGBoost"), ("ttml_rf", "random forest")):
    for metric, metric_name in (
        ("error_best", "Best error"),
        ("error_mean", "Mean error"),
    ):
        for threshold_method in ("forest", "data"):
            df_rf = df[df.estimator_name == model].copy()
            df_rf = df_rf[df_rf["threshold_method"] == threshold_method]
            df_rf["rank"] = df_rf.estimator_kwargs.map(rank_fun)
            df_rf["num_thresh"] = df_rf.estimator_kwargs.map(thresh_fun)

            plt.subplot(2, 2, i)
            gb = df_rf.groupby("rank")

            plt.plot(gb[metric].min(), label=threshold_method)
        xgb_best = np.min(df.loc[df["estimator_name"] == "xgb"][metric])
        plt.axhline(xgb_best, c="k", ls="--", label="xgboost")
        plt.axhline(
            df.loc[df["estimator_name"] == "rf"].iloc[0][metric],
            c="r",
            ls="--",
            label="random forest",
        )
        plt.legend()
        plt.title(f"Rank vs. {metric_name}, initialized with {model_name}")
        plt.xlabel("TT-rank")
        if task == "regression":
            plt.ylabel("MSE")
        else:
            plt.ylabel("Cross entropy")

        i += 1
# %%

# %%
# Inference speed vs. model complexity
df_ttml = df[df["estimator_name"].str.startswith("ttml_rf")].copy()


df_ttml["rank"] = df_ttml.estimator_kwargs.map(rank_fun)
df_ttml["num_thresh"] = df_ttml.estimator_kwargs.map(thresh_fun)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
ranks = df_ttml["rank"].unique()
for rank in ranks:
    if rank <= 11:
        marker = "+"
    else:
        marker = "x"
    df_rank = df_ttml[df_ttml["rank"] == rank]
    plt.plot(df_rank["num_params"], df_rank["speed"], marker=marker, linestyle='None')
plt.title("Inference speed vs. model complexity")
plt.xscale("log")
plt.xlabel("Number of parameters")
plt.ylabel("Samples / s")

# XGB and RF are kinda off-the-charts Let's just print info instead
xgb_sample = df[
    (df["estimator_name"] == "xgb") & (df["estimator_kwargs"] == "1024")
].iloc[0]
print(
    f"xgb num params: {xgb_sample.loc['num_params']:.4e}, xgb speed: {xgb_sample.loc['speed']:.4e}"
)
# plt.plot([xgb_sample.loc['num_params']],[xgb_sample.loc['speed']],'o')
rf_sample = df[(df["estimator_name"] == "rf")].iloc[0]
print(
    f"rf num params: {rf_sample.loc['num_params']:.4e}, rf speed: {rf_sample.loc['speed']:.4e}"
)
print(f"maximum ttml num params: {df_ttml['num_params'].max()}")
# plt.plot([rf_sample.loc['num_params']],[rf_sample.loc['speed']],'o')

plt.subplot(1, 2, 2)
import scipy.stats

exponent = 2
res = scipy.stats.linregress(df_ttml["rank"] ** (1 / exponent), df_ttml["speed"])
X_plot = np.unique(df_ttml["rank"] ** (1 / exponent))
plt.plot(X_plot ** exponent, res.slope * X_plot + res.intercept, c="k", alpha=0.5)
print(res)

for rank in ranks:
    if rank <= 11:
        marker = "+"
    else:
        marker = "x"
    df_rank = df_ttml[df_ttml["rank"] == rank]
    plt.plot(df_rank["rank"], df_rank["speed"], marker=marker, linestyle='None')
# plt.plot(df_ttml["rank"], df_ttml["speed"], ".")
plt.title("Inference speed vs. model complexity")
plt.xlabel("TT-rank")
plt.ylabel("Samples / s")
plt.xticks(np.arange(2, 20, 1))


plt.savefig(
    f"./figures/inference_speed.pdf",
    format="pdf",
    bbox_inches="tight",
)


# %%
# Let's try fitting a x**3 line to this.
df_ttml["num_thresh"].unique()

# %%
# Rank vs. loss with standard deviation
plt.figure(figsize=(14, 5))
num_thresh = 150
for i, model_name in enumerate(("ttml_rf", "ttml_xgb")):
    plt.subplot(1, 2, i + 1)
    df_rf = df[df.estimator_name == model_name].copy()
    df_rf["rank"] = df_rf.estimator_kwargs.map(rank_fun)
    df_rf["num_thresh"] = df_rf.estimator_kwargs.map(thresh_fun)
    df_rf = df_rf[df_rf["num_thresh"] == num_thresh].copy()

    plt.plot(df_rf["rank"], df_rf["error_mean"])
    plt.fill_between(
        df_rf["rank"],
        df_rf["error_mean"] - df_rf["error_std"],
        df_rf["error_mean"] + df_rf["error_std"],
        alpha=0.5,
    )

    xgb_best = np.min(df.loc[df["estimator_name"] == "xgb"][metric])
    plt.axhline(xgb_best, c="k", ls="--", label="xgboost (best)")
    plt.axhline(
        df.loc[df["estimator_name"] == "rf"].iloc[0][metric],
        c="r",
        ls="--",
        label="random forest",
    )

    plt.xlabel("Rank")
    plt.ylabel("MSE")
    plt.title(f"Rank vs. error, {num_thresh=}, {model_name=}")
# %%
DATASET_NAME = "concrete"
df2 = pd.read_csv("benchmark2.csv")
df2 = df2[df2["dataset_name"] == DATASET_NAME]
estimator = "ttml_rf"
plt.figure(figsize=(12, 10))
for i, estimator in enumerate(["ttml_rf", "ttml_xgb", "xgb"]):
    plt.subplot(2, 2, i + 1)
    trials = df2[df2["estimator_name"] == estimator]["individual_trials"].iloc[
        -2
    ]
    trials = np.array(eval(trials))
    plt.plot(((trials)))
    plt.title(estimator)
