# %%
"""Simple script to run benchmarks.py for a bunch of parameters"""

from benchmarks import do_benchmark
import benchmarks
import numpy as np
from itertools import product

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

for DATASET_NAME in regression_datasets + categorical_datasets:
    seed = np.random.randint(1000)

    print(f"----------> {DATASET_NAME} rf")
    do_benchmark(DATASET_NAME, "rf", seed=seed)
    for i in range(11):
        num_boost_round = 2 ** i
        print(f"------------> xgb {DATASET_NAME} {i}/10")
        do_benchmark(
            DATASET_NAME,
            "xgb",
            seed=seed,
            estimator_kwargs=str(num_boost_round),
        )

    if DATASET_NAME == "airfoil":
        num_thresholds = [15, 25, 50, 100, 150]
    elif DATASET_NAME == "concrete":
        num_thresholds = [15, 20, 25, 35, 50]
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
            print(
                f"----------> {len(estimators)*i+j+1}/{len(all_args)*len(estimators)}, {DATASET_NAME} {estim}-{arg} {threshold_method}"
            )
            do_benchmark(DATASET_NAME, estim, estimator_kwargs=arg, seed=seed)
