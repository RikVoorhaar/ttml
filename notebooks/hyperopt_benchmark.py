"""Script for performing hyperoptimization on a particular dataset for a
particular estimator. Used to determine best possible performance, for better
comparison between estiamtors."""

import argparse

import os

from benchmarks import do_benchmark
import numpy as np
import benchmarks
import datetime


def dt():
    return datetime.datetime.now()


benchmarks.BENCHMARK_CSV = "results/hyperopt_results.csv"


def run_hyperopt(dataset_name, optimizer, num_its):
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    import pickle

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
        kwarg = f"{int(params['r'])}-{int(params['num_thresh'])}"
        print("\n", "+" * 75)
        print(f"{dt()}  START: {dataset_name}-{optimizer}-{kwarg}")
        print("+" * 75, "\n")
        output = do_benchmark(
            dataset_name, optimizer, estimator_kwargs=kwarg, force=True
        )
        (error_mean, error_std, _, num_params, _, _, errors) = output

        return {
            "loss": error_mean,
            "status": STATUS_OK,
            "loss_variance": error_std / np.sqrt(len(errors) - 1),
            "attachments": {"num_params": num_params, "errors": str(errors)},
        }

    space = {
        "r": hp.quniform("r", 2, 21, 1),
        "num_thresh": hp.quniform("num_thresh", 5, 200, 1),
    }
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do hyperparamater optimization for benchmarking."
    )
    parser.add_argument(
        "dataset_name",
        metavar="name",
        nargs="?",
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "optimizer",
        nargs="?",
        type=str,
        help="name of optimizer (e.g. 'ttml_xgb')",
    )
    parser.add_argument(
        "num_its",
        metavar="n_its",
        nargs="?",
        type=int,
        help="number of iterations of hyperoptimization",
    )

    args = parser.parse_args()
    if args.dataset_name is None:
        print("error: no dataset name suplied")
        parser.print_help()
    if args.optimizer is None:
        print("error: no optimizer name suplied")
        parser.print_help()
    elif args.num_its is None:
        print("error: no number of iterations supplied")
        parser.print_help()
    else:
        run_hyperopt(
            args.dataset_name,
            args.optimizer,
            args.num_its,
        )
