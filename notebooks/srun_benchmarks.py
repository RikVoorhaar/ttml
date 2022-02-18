# %%
"""Simple script to run benchmarks.py with a slurm job array for a bunch of
parameters"""

from benchmarks import do_benchmark
import benchmarks
import numpy as np
from itertools import product
import os
import datetime
import pathlib
import pandas as pd

benchmarks.BENCHMARK_CSV = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "results", "benchmark_all.csv"
)

categorical_datasets = [
    # "seismic_bumps",
    # "ai4i2020",
    # "diabetic_retinopathy",
    # "bank_marketing",
    # "default_credit_card",
    # "census_income",
    # "electrical_grid",
    "shill_bidding",
    # "vehicle_coupon",
    # "bankruptcy",
    # "mini_boone",
]

regression_datasets = [
    "airfoil",
    "concrete",
    # "power_plant",
    # "gas_turbine",
    # "wine_quality",
]

# good_datasets = ("airfoil,concrete,shill_bidding")
estimators = ["ttml_rf", "ttml_xgb", "ttml_mlp1", "ttml_mlp2", "ttml_gp"]
all_jobs = []

for DATASET_NAME in regression_datasets + categorical_datasets:
    seed = 179
    all_jobs.append((DATASET_NAME, "rf", None, seed))
    for i in range(11):
        num_boost_round = 2 ** i
        all_jobs.append((DATASET_NAME, "xgb", str(num_boost_round), seed))

    if DATASET_NAME == "airfoil":
        num_thresholds = [15, 25, 50, 100, 150]
    elif DATASET_NAME == "concrete":
        num_thresholds = [15, 20, 25, 30, 35, 50]
    elif DATASET_NAME == "shill_bidding":
        num_thresholds = [15, 20, 25, 35, 50]
    else:
        num_thresholds = [15, 20, 25, 35, 40, 50, 60, 70, 80, 90, 100, 120, 150]
    ranks = range(2, 20)
    all_args = list(
        product(ranks, num_thresholds, ["data"])
    )  # just use 'data' method
    for i, (r, num_thresh, threshold_method) in enumerate(all_args):
        benchmarks.THRESHOLD_METHOD = threshold_method
        arg = f"{r}-{num_thresh}"
        for j, estim in enumerate(estimators):
            all_jobs.append((DATASET_NAME, estim, arg, seed))
print(f"{datetime.datetime.now()}  {len(all_jobs)} jobs generated")
# %%

# prune jobs that have already been done

BENCHMARK_CSV = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "results", "benchmark_all.csv"
)


def check_if_done(dataset_name, estimator_name, estimator_kwargs):
    selector = (results["dataset_name"] == dataset_name) & (
        results["estimator_name"] == estimator_name
    )
    if estimator_kwargs is not None:
        selector = selector & (results["estimator_kwargs"] == estimator_kwargs)
    return sum(selector) > 0


if os.path.isfile(BENCHMARK_CSV):
    results = pd.read_csv(BENCHMARK_CSV)
    new_jobs = []
    for job in all_jobs:
        dataset_name, estimator_name, estimator_kwargs, _ = job
        if not check_if_done(dataset_name, estimator_name, estimator_kwargs):
            new_jobs.append(job)
    print(
        f"{datetime.datetime.now()}  After pruning {len(new_jobs)}/{len(all_jobs)} jobs remaining"
    )
    all_jobs = new_jobs
else:
    print("No previous jobs found, doing everything")

all_jobs
# %%
try:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
except (KeyError, ValueError):
    task_id = 0
    num_tasks = 1

print(f"{task_id=}")
MAX_JOBS_PER_TASK = 100
num_jobs = len(all_jobs)
jobs_per_task = min(MAX_JOBS_PER_TASK, num_jobs // num_tasks)
print(f"I was assigned {jobs_per_task} jobs")
print(f"Doing jobs {jobs_per_task*task_id}-{jobs_per_task*(task_id+1)-1}")

for job in all_jobs[
    jobs_per_task * task_id : jobs_per_task * (task_id + 1) - 1
]:
    print("\n")
    print("-" * 50)
    print(f"{datetime.datetime.now()}  Starting job with parameters {job}")
    print("-" * 50)

    dataset_name, estimator_name, estimator_kwargs, seed = job
    do_benchmark(
        dataset_name,
        estimator_name,
        estimator_kwargs=estimator_kwargs,
        seed=seed,
    )

    print("\n")
    print("-" * 50)
    print(
        f"{datetime.datetime.now()}  Sucessfully completed job with parameters {job}"
    )
    print("-" * 50)
