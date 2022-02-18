from copy import copy
from functools import reduce
import operator
import warnings

import autoray as ar
import numpy as np
import scipy.special

try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass
try:
    import torch
except ModuleNotFoundError:
    pass
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer


def convert_backend_cores(cores, backend):
    """Convert the backend of a list of cores to target backend."""
    return [convert_backend(C, backend) for C in cores]


def convert_backend(A, backend):
    """Convert the backend of tensor to target backend."""
    if ar.infer_backend(A) == backend:
        return A
    elif backend == "numpy":
        return ar.to_numpy(A)
    else:
        if ar.infer_backend(A) != "numpy":
            A = ar.to_numpy(A)
        if backend == "tensorflow":
            return tf.constant(A)
        elif backend == "torch":
            return torch.from_numpy(A)
        else:
            ar.do("array", A, like=backend)


def random_idx(tt, N, backend=None):
    """Generate `N` random indices for the tensor train `tt`"""
    if backend is None:
        backend = tt.backend
    idx = np.stack([np.random.choice(d, size=N) for d in tt.dims], axis=-1)
    return convert_backend(idx, backend)


def random_normal(size, backend="numpy"):
    """Generate float64 standard normal distribution of specified size

    TODO: This is unnecessary since the `with_dtype` wrapper update in autoray
    """
    if backend in ("numpy", "dask", "sparse"):
        A = ar.do("random.normal", size=size, like=backend)
    else:
        A = ar.do(
            "random.normal",
            size=size,
            like=backend,
            dtype=ar.to_backend_dtype("float64", backend),
        )
    return A


def random_isometry(size, backend="numpy", dtype="float64"):
    A = ar.do("random.normal", size=size, like=backend, dtype=dtype)
    Q, _ = ar.do("linalg.qr", A)
    return Q


def merge_sum(idx, y, backend=None):
    """Merge entries of y with identical entry in idx and sum result.

    Returns new indices, merged y. This is copypasta from stackoverflow user
    perimosocordiae."""
    if backend is None:
        backend = ar.infer_backend(idx)
    idx = convert_backend(idx, "numpy")
    y = convert_backend(y, "numpy")

    order = np.lexsort(idx.T)
    diff = np.diff(idx[order], axis=0)
    uniq_mask = np.append(True, (diff != 0).any(axis=1))
    inv_idx = np.zeros_like(order)
    inv_idx[order] = np.cumsum(uniq_mask) - 1
    return convert_backend(idx[order][uniq_mask], backend), convert_backend(
        np.bincount(inv_idx, weights=y), backend
    )


def project_sorted(a, v):
    """
    Returns closest entry in sorted array `a` for each entry in `v`
    """

    inds = np.minimum(np.searchsorted(a, v, side="right"), len(a) - 1)
    inds_min_one = np.maximum(inds - 1, 0)
    first_better = np.abs(a[inds] - v) <= np.abs(a[inds_min_one] - v)
    projected = a[inds_min_one]
    projected[first_better] = a[inds[first_better]]

    return projected


def univariate_kmeans(X, n_clusters, prune_factor=10):
    """
    Use k-means to find a set of `n_clusters` points minimizing the average
    minimum distance of X to the set.

    prune_factor determines the minimum number of points each centroid should be
    associated to. A prune_factor of 10 means that centroids are ignored with
    size less than 1/10 times the average cluster size.
    """
    warnings.warn(
        "Don't use this, better use `KBinsDiscretizer`", DeprecationWarning
    )
    # number of clusters should never exceed number of different data points
    n_clusters = min(n_clusters, len(np.unique(X)))

    # Initialize k-means with percentiles for speedup
    cluster_init = np.percentile(X, np.linspace(0, 100, n_clusters)).reshape(
        -1, 1
    )

    # jitter to avoid duplicate points
    eps = (cluster_init[-1] - cluster_init[0]) * 1e-8
    cluster_init = cluster_init + np.random.uniform(
        eps, size=cluster_init.shape
    )

    km = KMeans(
        n_clusters=n_clusters,
        n_init=1,
        init=cluster_init,
        max_iter=50,
        tol=1e-3,
    )
    # sklearn likes to complain about insufficient number of clusters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km.fit(X.reshape(-1, 1))

    # Throw away points with too few associated samples
    centers = km.cluster_centers_.reshape(-1)
    if prune_factor is not None:
        min_num = len(X) // (prune_factor * n_clusters)
        labels = np.concatenate(
            [km.labels_, np.arange(0, n_clusters, dtype=int)]
        )  # add entire range 0,...,N to make bincount work as expected
        centers = centers[np.bincount(labels) > min_num]
    return np.sort(centers)


def add_infty_bins(thresholds, replace_biggest=False):
    """For a list of arrays of thresholds, add `np.infty` to the end of every
    array. If `replace_biggest=True`, replace the last element by `np.infty`
    instead."""
    new_thresholds = []
    for thresh_list in thresholds:
        new_list = np.copy(thresh_list)
        if replace_biggest:
            new_list[-1] = np.infty
        else:
            new_list = np.concatenate([new_list, [np.infty]])
        new_thresholds.append(new_list)
    return new_thresholds


def thresholds_from_data(X, num_thresh, min_samples=5, strategy="quantile"):
    """Bin each feature in at most 'num_thresh' bins. Compresses bins such that
    each bin contains at least `min_samples` samples."""
    num_thresh_feat = np.array([num_thresh] * X.shape[1])
    check_sum = -1
    while np.sum(num_thresh_feat) != check_sum:
        check_sum = np.sum(num_thresh_feat)
        discretizer = KBinsDiscretizer(
            n_bins=num_thresh_feat, strategy=strategy, encode="ordinal"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X_transformed = discretizer.fit_transform(X)
            except ValueError:
                raise ValueError(f"Invalid number of bins: {num_thresh_feat}")
        for i in range(X.shape[1]):
            counts = np.bincount(
                X_transformed[:, i].astype(int), minlength=num_thresh_feat[i]
            )
            if np.min(counts) < min_samples:
                num_reduce = max(1, np.sum(counts < min_samples) // 2)
                num_thresh_feat[i] = max(2, num_thresh_feat[i] - num_reduce)
    return add_infty_bins(discretizer.bin_edges_)


def predict_logit(logits, random=False):
    """Turns logits into 0 / 1 predictions.

    If `random=True` then sample from Bernouilli"""

    probs = ar.do("sigmoid", logits)
    if random:
        preds = probs > ar.do("random.uniform", size=len(probs), like=probs)
    else:
        preds = probs > 0.5
    preds = ar.astype(preds, "float64")
    return preds


def trim_ranks(dims, ranks):
    """Return TT-rank to which TT can be exactly reduced

    A tt-rank can never be more than the product of the dimensions on the left
    or right of the rank. Furthermore, any internal edge in the TT cannot have
    rank higher than the product of any two connected supercores. Ranks are
    iteratively reduced  for each edge to satisfy these two requirements until
    the requirements are all satisfied.
    """
    ranks = list(ranks)

    for i, r in enumerate(ranks):
        dim_left = reduce(operator.mul, dims[: i + 1], 1)
        dim_right = reduce(operator.mul, dims[i + 1 :], 1)
        # dim_left = np.prod()
        # dim_right = np.prod(dims[i + 1 :])
        ranks[i] = min(r, dim_left, dim_right)
    changed = True
    ranks = [1] + ranks + [1]
    while changed:
        changed = False
        for i, d in enumerate(dims):
            if ranks[i + 1] > ranks[i] * d:
                changed = True
                ranks[i + 1] = ranks[i] * d
            if ranks[i] > d * ranks[i + 1]:
                changed = True
                ranks[i] = d * ranks[i + 1]

    return tuple(ranks[1:-1])


def matricize(A, mode):
    """Matricize tensor ``A`` with respect to ``mode``"""
    if isinstance(mode, int):
        mode = (mode,)
    perm = mode + tuple(i for i in range(len(A.shape)) if i not in mode)
    A = np.transpose(A, perm)
    A = A.reshape(A.shape[: len(mode)] + (np.prod(A.shape[len(mode) :]),))
    return A


def dematricize(A, mode, shape):
    """Undo matricization of ``A`` with respect to ``mode``. Needs ``shape`` of
    original tensor."""
    current_shape = [A.shape[0]] + [s for i, s in enumerate(shape) if i != mode]
    current_shape = tuple(current_shape)
    A = A.reshape(current_shape)
    perm = list(range(1, len(shape)))
    perm = perm[:mode] + [0] + perm[mode:]
    A = np.transpose(A, perm)
    return A


# ----------------------------
# Autoray function definitions

SUPPORTED_BACKENDS = ["numpy", "torch", "tensorflow"]


def tf_dot(x, y):
    return tf.tensordot(x, y, 1)


ar.register_function("tensorflow", "dot", tf_dot)
ar.register_function("numpy", "sigmoid", scipy.special.expit)
ar.register_function("numpy", "logit", scipy.special.logit)