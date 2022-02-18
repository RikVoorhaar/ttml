"""Scheduling script for hyperparameter optimization. To be run on a SLURM job array"""
from hyperopt_benchmark import run_hyperopt
import pandas as pd
import datetime
import numpy as np
import os

TARGET_ITS = 20


def dt():
    return datetime.datetime.now()


datasets = [
    "seismic_bumps",
    "ai4i2020",
    "diabetic_retinopathy",
    "bank_marketing",
    "default_credit_card",
    "census_income",
    "electrical_grid",
    "shill_bidding",
    "online_shoppers",
    # "vehicle_coupon",
    # "bankruptcy",
    # "mini_boone",
    "airfoil",
    "concrete",
    "power_plant",
    "gas_turbine",
    "wine_quality",
]

estimators = [
    "ttml_xgb",
    "ttml_rf",
    "ttml_mlp1",
    "ttml_mlp2",
]

all_jobs = np.indices([len(datasets), len(estimators)])
all_jobs = all_jobs.T.reshape(-1, 2)
np.random.seed(179)
np.random.shuffle(all_jobs)  # Shuffle to create more equal jobs

pruned_jobs = []
results_raw = pd.read_csv("results/hyperopt_results.csv")
for job in all_jobs:
    dataset_name = datasets[job[0]]
    estimator_name = estimators[job[1]]

    job_results = results_raw[
        (results_raw["dataset_name"] == dataset_name)
        & (results_raw["estimator_name"] == estimator_name)
    ]
    if len(job_results) < TARGET_ITS:
        pruned_jobs.append(job)

print(f"{dt()}  {len(pruned_jobs)}/{len(all_jobs)} remaining after pruning")
all_jobs = pruned_jobs

try:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
except (KeyError, ValueError):
    task_id = 0
    num_tasks = 1
jobs_per_task = len(all_jobs) // num_tasks
my_jobs = all_jobs[jobs_per_task * task_id : jobs_per_task * (task_id + 1)]
print(f"{dt()}  I got assigned {len(my_jobs)} jobs\n\n")

current_target = TARGET_ITS
while current_target <= 200:
    for job in my_jobs:
        dataset_name = datasets[job[0]]
        estimator = estimators[job[1]]
        print(f"{dt()}  Doing job {dataset_name}-{estimator}")
        print("=" * 80)
        run_hyperopt(dataset_name, estimator, current_target)
    # After we did 20 its on all jobs, just increase the number of its to
    # maybe get better results. This way job is probably used entire 12h.
    current_target += 10

# %%
[(datasets[job[0]], estimators[job[1]]) for job in all_jobs]