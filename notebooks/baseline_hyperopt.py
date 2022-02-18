# %%
"""Hyperoptimization script and scheduler for running baseline benchmarks"""


from benchmarks import do_benchmark
import benchmarks
import pandas as pd
import datetime
import pickle
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os

TARGET_ITS = 20
MAX_ITS = 50
benchmarks.BENCHMARK_CSV = "results/hyperopt_results.csv"


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
    "airfoil",
    "concrete",
    "power_plant",
    "gas_turbine",
    "wine_quality",
]

hyper_param_space = {
    "xgb": {
        "eta": hp.loguniform("eta", np.log(1e-3), 0),
        "max_depth": hp.quniform("max_depth", 5, 30, 1),
        "n_estimators": hp.qloguniform(
            "n_estimators", np.log(50), np.log(1000), 1
        ),
    },
    "rf": {
        "max_leaf_nodes": hp.qloguniform(
            "max_leaf_nodes", np.log(50), np.log(2000), 1
        ),
        "max_depth": hp.qloguniform("max_depth", np.log(5), np.log(100), 1),
        "n_estimators": hp.qloguniform(
            "n_estimators", np.log(50), np.log(1000), 1
        ),
    },
}


def run_hyperopt(dataset_name, optimizer, num_its):
    # Load previous work
    pickle_file = os.path.join(
        "param_pickles", f"{dataset_name}_{optimizer}.pkl"
    )
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            trials = pickle.load(f)
        print(f"Continuing from previous state, done {len(trials)} iterations.")
    else:
        trials = Trials()
        with open(pickle_file, "wb") as f:
            pickle.dump(trials, f)

    def objective(params):
        print("\n", "+" * 75)
        print(f"{dt()}  START: {dataset_name}-{params}")
        print("+" * 75, "\n")
        params_int = dict()
        for k, v in params.items():
            if abs(int(v) - v) < 1e-8:
                params_int[k] = int(v)
            else:
                params_int[k] = v
        output = do_benchmark(
            dataset_name,
            optimizer,
            estimator_hyperparams=params_int,
            force=True,
        )
        (error_mean, error_std, _, num_params, _, _, errors) = output

        return {
            "loss": error_mean,
            "status": STATUS_OK,
            "loss_variance": error_std / np.sqrt(len(errors) - 1),
            "attachments": {"num_params": num_params, "errors": str(errors)},
        }

    space = hyper_param_space[optimizer]
    iterator = range(len(trials) + 1, num_its + 1)
    for n_trials in iterator:
        fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            show_progressbar=False,
        )
        with open(pickle_file, "wb") as f:
            pickle.dump(trials, f)
        print("\n", "~" * 75)
        print(f"{dt()}  Hyperoptimization for {dataset_name}-{optimizer}")
        print(f"{n_trials}/{num_its}")
        print(f"Best loss: {np.min(trials.losses()):.5e}")
        print("~" * 75, "\n")


def get_jobs():
    estimators = list(hyper_param_space.keys())
    all_jobs = np.indices([len(datasets), len(estimators)])
    all_jobs = all_jobs.T.reshape(-1, 2)

    pruned_jobs = []
    results_raw = pd.read_csv("results/hyperopt_results.csv")
    for job in all_jobs:
        dataset_name = datasets[job[0]]
        estimator_name = estimators[job[1]]

        job_results = results_raw[
            (results_raw["dataset_name"] == dataset_name)
            & (results_raw["estimator_name"] == estimator_name)
        ]
        if len(job_results) < MAX_ITS:
            pruned_jobs.append((dataset_name, estimator_name))
    print(f"{dt()}  {len(pruned_jobs)}/{len(all_jobs)} remaining after pruning")
    return pruned_jobs


current_target = TARGET_ITS
if __name__ == "__main__":
    while current_target <= MAX_ITS:
        try:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        except (KeyError, ValueError):
            task_id = 0
            num_tasks = 1

        all_jobs = get_jobs()
        print(f"{task_id=}")
        MAX_JOBS_PER_TASK = 100
        num_jobs = len(all_jobs)
        jobs_per_task = min(MAX_JOBS_PER_TASK, num_jobs // num_tasks)
        my_jobs = all_jobs[
            jobs_per_task * task_id : jobs_per_task * (task_id + 1)
        ]
        print(f"{dt()}  I got assigned {len(my_jobs)} jobs\n\n")

        for job in my_jobs:
            print("\n")
            print("-" * 50)
            print(f"{dt()}  Starting job with parameters {job}")
            print("-" * 50)

            dataset_name, estimator = job
            print(f"{dt()}  Doing job {dataset_name}-{estimator}")
            print("=" * 80)
            run_hyperopt(dataset_name, estimator, current_target)
        current_target += 10

# %%
