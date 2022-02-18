# %%
"""Script for analysing effect of permuting features on final test error for ttml."""

from itertools import permutations
from math import perm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

permutation_df = pd.read_csv("results/permutation_airfoil.csv")
# %%
plt.figure(figsize=(8, 5))
error_mean = np.array(permutation_df["error_mean"])
error_mean.sort()
plt.plot(error_mean)
plt.xlabel("Rank of permutation")
plt.ylabel("MSE")
plt.show()

# %%
"""We estimate the standard deviation for each run by first subtracting the
mean of each trial. As we see below, all these standard deviations seem to
come from the same distribution, so we replace them by a (normalized) average.
"""

plt.figure(figsize=(8, 5))
trials = []
for trial in permutation_df["individual_trials"]:
    trials.append(eval(trial))
trials = np.array(trials)
trials_corrected = trials - np.mean(trials, axis=0)
trials_std = np.std(trials_corrected, axis=1)
print("sigma0=",np.mean(trials_std/error_mean) / np.sqrt(trials.shape[1] - 1))
trials_std = np.mean(trials_std) * error_mean / np.mean(error_mean)
trials_std_norm = trials_std / np.sqrt(trials.shape[1] - 1)

error_mean = np.array(permutation_df["error_mean"])
order = np.argsort(error_mean)
# plt.fill_between(
#     np.arange(len(error_mean)),
#     error_mean[order] - trials_std[order],
#     error_mean[order] + trials_std[order],
#     alpha=0.5,
# )
plt.fill_between(
    np.arange(len(error_mean)),
    error_mean[order] - trials_std_norm[order],
    error_mean[order] + trials_std_norm[order],
    alpha=0.5,
)
plt.plot(error_mean[order])
plt.xlabel("Rank of permutation")
plt.ylabel("MSE")
plt.savefig(
    f"./figures/permutation_effect.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
# %%
"""As a test, we can see if all the standard deviations of each run belong
to the same distribution. We simply normalize them, and then compute their
p-values assuming they are drawn from the same t-distribution. The result
seems to indicate that they come from the same distribution, so in the plot
above we can replace the stds with a normalized version."""
import scipy.stats

trials_std = np.std(trials_corrected, axis=1)
trials_std *= error_mean / np.mean(error_mean)
mean_std = np.mean(trials_std)
std_std = np.std(trials_std)  # / np.sqrt(len(trials_std)-1)
t = (trials_std - mean_std) / std_std
sorted_p = np.sort(2 * (1 - scipy.stats.t.cdf(abs(t), len(trials_std) - 1)))
plt.plot(sorted_p)
plt.plot(np.linspace(0, 1, len(trials_std)))
plt.show()

std_std

# %%
"""Let's look for patterns in the permutations. Only thing we see is that
feature 0 is best put in position 0 or 4, nothing else is significant."""
permutation_df_sorted = permutation_df.copy()
permutation_df_sorted.sort_values("error_mean", axis=0, inplace=True)
error_mean_sorted = np.array(permutation_df_sorted["error_mean"])
permutations = []
for perm in permutation_df_sorted["permutation"]:
    permutations.append(eval(perm))

permutations = np.array(permutations)
best_positions = np.argsort(permutations, axis=1)
best_positions = np.min([best_positions, 4 - best_positions], axis=0)
means = [
    [error_mean[best_positions[:, j] == i] for i in [0, 1, 2]] for j in range(5)
]
# scipy.stats.ttest_ind(means[0],means[1])[1]
pvals = [
    [
        scipy.stats.ttest_ind(means[j][i1], means[j][i2])[1]
        for i1, i2 in [(0, 1), (0, 2), (1, 2)]
    ]
    for j in range(5)
]
pvals

# %%
import sys
sys.path.insert(1, "..")
from datasets import load_data

dataset_name = "airfoil"
DATASET_FOLDER = "../datasets/data"
dataset = load_data.dataset_loaders[dataset_name](DATASET_FOLDER)
X_data = pd.DataFrame(dataset['X'])
[len(X_data[i].unique()) for i in range(5)]
# X_data.describe()
[np.corrcoef(X_data[i],dataset['y'])[0,1] for i in range(5)]
# %%
