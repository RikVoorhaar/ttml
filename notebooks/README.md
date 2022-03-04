This folder contains a number of useful scripts for running experiments using this library, as well as some
tutorials. In alphabetical order we list the purpose of each file.

In many scripts we use ``hyperopt`` for hyper-parameter optimization. 
Some scripts are specifically designed to be run on a `slurm` cluster, but can be easily adapted for different types of computational clusters.

- ``baseline_hyperopt.py`` Hyperoptimization script and scheduler for running baseline benchmarks
- ``benchmarks.py`` Base script for benchmarking and comparing different estimators. Has CLI support.
- ``Comparison_Linesearch_methods.ipynb`` Some test to compare different line search methods for RCGD
- ``complexity_vs_performance.py`` Script to make plots comparing the storage complexity to performance of different estimators
- ``forest_compression.py`` Looks at the effect of compression the number of thresholds in a random forest. This can be useful to represent random forests by low-rank tensors directly.
- ``hyperopt_anlysis.py`` Script for plotting and analysing the results of the ``hyperopt_benchmark.py``
- ``hyperopt_benchmark.py`` Script for performing hyperoptimization on a particular dataset for a particular
estimator. Used to determine best possible performance, for better comparison between estiamtors.
- ``hyperopt_scheduler.py`` Scheduling script for hyperparameter optimization. To be run on a SLURM job array
- ``optim_tut.ipynb`` Tutorial notebook for optimizing tensor trains using RCGD and line search. Also shown in the docs.
- ``permutation_analysis.py`` Script for investigating the effect of permuting features of the data on the final test error of the TTML estimator. This makes plots based on data produced by ``permutation_experiment.py``
- ``permutation_experiment.py`` Script for investigating the effect of permuting features of the data on the final test error of the TTML estimator. This provides the data that is then analyzed by ``permutation_analysis.py``
- ``plot_benchmarks.py`` Script for plotting the results of benchmark scripts. Used for several plots in the preprint.
- ``run_benchmarks.py`` Simple script to run ``benchmarks.py`` for a bunch of parameters
- ``srun_benchmarks.py`` Simple script to run ``benchmarks.py`` with a slurm job array for a bunch of parameters
- ``threshold_method_comparison.py`` Script for comparing the effect of using different discretizations other than the default equal frequency binning.
