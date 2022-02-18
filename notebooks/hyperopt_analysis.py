# %%
"""Script for plotting and analysing the results of the `hyperopt_benchmark.py`"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets_names = {
    "ai4i2020": "AI4I 2020",
    "airfoil": "Airfoil Self-noise",
    "bank_marketing": "Bank Marketing",
    "census_income": "Adult",
    "concrete": "Concrete compressive strength",
    "default_credit_card": "Default of credit card",
    "diabetic_retinopathy": "Diabetic Retinopathy Debrecen",
    "electrical_grid": "Electrical Grid Stability",
    "gas_turbine": "Gas Turbine Emissions",
    "online_shoppers": "Online Shoppers",
    "power_plant": "Combined Cycle Power Plant",
    "seismic_bumps": "Seismic-bumps",
    "shill_bidding": "Shill Bidding",
    "wine_quality": "Wine quality",
}
datasets = list(datasets_names.keys())


estimators = ["ttml_xgb", "ttml_rf", "ttml_mlp1", "ttml_mlp2", "xgb", "rf"]

all_jobs = np.indices([len(datasets), len(estimators)])
all_jobs = all_jobs.T.reshape(-1, 2)

# %%
results_raw = pd.read_csv("results/hyperopt_results.csv")
new_rows = []
for job in all_jobs:
    dataset_name = datasets[job[0]]
    estimator_name = estimators[job[1]]

    job_results = results_raw[
        (results_raw["dataset_name"] == dataset_name)
        & (results_raw["estimator_name"] == estimator_name)
    ]
    if len(job_results) == 0:
        print(f"No results for {dataset_name}-{estimator_name}")
        new_rows.append([dataset_name, estimator_name, np.NAN, np.NAN, 0])
        continue

    best_result = job_results["error_mean"].argmin()
    mean = job_results["error_mean"].iloc[best_result]
    std = job_results["error_std"].iloc[best_result]
    new_rows.append([dataset_name, estimator_name, mean, std, len(job_results)])
results_raw[results_raw['estimator_name']=='rf']
# %%
results = pd.DataFrame(
    new_rows,
    columns=["dataset_name", "estimator_name", "mean", "std", "num_tries"],
)
# results.sort_values(, inplace=True)
results.sort_values(["dataset_name", "estimator_name"], inplace=True)
results[results['num_tries']<20]


# %%


def format_num(num, std):
    num_digits = -int(np.floor(np.log10(std))) + 1

    std_norm = round(std * 10 ** num_digits)
    if num_digits == 1:
        std_norm /= 10
    s = f"{{num:.{num_digits}f}}({{std_norm}})"
    # s = f"{{num:.{num_digits}f}}+/- {{std:.{num_digits}f}}"
    s = s.format(num=num, std_norm=std_norm)
    return s


columns = ["Dataset"]+["task","#feat.",'#samples']+estimators
results_table = pd.DataFrame(index=datasets, columns=columns)
for row in results.iterrows():
    estimator_name = row[1].loc["estimator_name"]
    dataset_name = row[1].loc["dataset_name"]
    mean = row[1].loc["mean"]
    std = row[1].loc["std"]
    results_table.loc[dataset_name][estimator_name] = format_num(mean, std)

from datasets.load_data import dataset_loaders

DATASET_FOLDER = "../datasets/data/"


for dataset_name in datasets:
    results_table.loc[dataset_name]['Dataset'] = datasets_names[dataset_name]
    data = dataset_loaders[dataset_name](DATASET_FOLDER)
    if data['regression']:
        results_table.loc[dataset_name]['task'] = 'Regr.'
    else:
        results_table.loc[dataset_name]['task'] = 'Class.'
    
    results_table.loc[dataset_name]['#feat.'] = data['X'].shape[1]
    results_table.loc[dataset_name]['#samples'] = data['X'].shape[0]

print(results_table.to_latex(index=False,columns=["Dataset","task","#feat.","#samples"]))
# %%
print(results_table.to_latex(index=False,columns=["Dataset"]+estimators))
# %%
