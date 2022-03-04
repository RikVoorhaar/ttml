"""Implements conversion of random forests to compressed tensors"""

from copy import deepcopy

import numpy as np
from scipy.linalg import svd
from sklearn.tree._classes import BaseDecisionTree
from sklearn.utils.extmath import randomized_svd

from ttml.tensor_train import TensorTrain
from ttml.utils import project_sorted, univariate_kmeans


def compress_forest_thresholds(
    forest,
    num_thresh,
    X=None,
    project_thresh_onto_X=False,
    inplace=False,
    use_univariate_kmeans=True,
):
    """
    Compress the total number of thresholds used in a forest.

    Parameters
    ----------
    forest:
        Random forest to compress. Can be any class with an `estimators_`
        attribute containing a list of scikit-learn `DecisionTreeRegressor` or
        `DecisionTreeClassifier` objects. This essentially includes all classes
        of sklearn.ensemble
    num_thresh: int
        Maximum number of thresholds per feature
        #TODO: Allow different number of thresholds for each feature.
    X: np.ndarray (optional)
        Data to sample thresholds from if `project_thresh_onto_X=True`
    project_thresh_onto_X: bool (default: False)
        Project thresholds onto data values occuring in `X`. This may degrade
        performance.
    inplace: bool (default: False)
        Compress the forest inplace if True, compress a copy of the forest
        otherwise.
    use_univariate_kmeans: bool (default: True)
        Use univariate kmeans clustering to extract thresholds. If False, use
        percentile statistics instead. The former is slower, but tends to give
        better thresholds.

    Returns
    -------
    compressed_forest: same type as input `forest`
    compressed_thresholds: list<np.ndarray>
        List of thresholds for each feature.
    """
    # Gather all the unique values occuring in the data X for each feature
    # Also include the midpoints between consecutive values
    if X is not None and project_thresh_onto_X:
        X_unique = [np.unique(feat) for feat in X.T]
        X_unique = [
            np.concatenate([feat, (feat[1:] + feat[:-1]) / 2])
            for feat in X_unique
        ]
        X_unique = [np.unique(feat) for feat in X_unique]

    if not inplace:  # Make copy if not inplace
        forest = deepcopy(forest)

    # List all thresholds occuring in the forest for each feature
    thresholds = [[] for _ in range(forest.n_features_in_)]
    for tree_estim in forest.estimators_:
        if isinstance(tree_estim, np.ndarray):
            tree_estim = tree_estim[0]
        tree = tree_estim.tree_
        for feat in range(forest.n_features_in_):
            # Find indices of nodes corresponding to a split for feature `feat`
            (inds,) = np.where(tree.feature == feat)
            projected_thresh = tree.threshold[inds]
            if project_thresh_onto_X:  # Project thresholds onto data values
                project_sorted(X[feat], projected_thresh)
            thresholds[feat].append(projected_thresh)

    # For each feature, find a representative set of thresholds. Then project all thresholds onto
    # this representative set.
    new_thresholds = []
    compressed_thresholds = []
    for i, threshold_list in enumerate(thresholds):
        if use_univariate_kmeans:
            compressed_thresholds.append(
                univariate_kmeans(np.concatenate(threshold_list), num_thresh)
            )
        else:
            new_compressed = np.concatenate(threshold_list)
            new_compressed = new_compressed[
                new_compressed < np.max(new_compressed)
            ]
            percents = np.arange(1, num_thresh) * 100 / num_thresh
            new_compressed = np.percentile(
                new_compressed, percents, interpolation="nearest"
            )
            new_compressed = np.unique(new_compressed)
            compressed_thresholds.append(new_compressed)
        new_thresholds.append(
            [
                project_sorted(compressed_thresholds[i], t)
                for t in threshold_list
            ]
        )
    # add infinity too all of the thresholds, needed for compatibility with TTML
    compressed_thresholds = [
        np.concatenate([t, [np.infty]]) for t in compressed_thresholds
    ]

    # Update the thresholds to the new values for each tree
    for i, tree in enumerate(forest.estimators_):
        if isinstance(tree, np.ndarray):
            tree = tree[0]
        tree = tree.tree_

        # tree.threshold is readonly, need to use __setstate__ to change it
        tree_thresholds = np.copy(tree.threshold)
        for feat in range(forest.n_features_in_):
            (inds,) = np.where(tree.feature == feat)
            tree_thresholds[inds] = new_thresholds[feat][i]
        state = tree.__getstate__()
        state["nodes"]["threshold"] = tree_thresholds
        tree.__setstate__(state)
    return forest, compressed_thresholds


def tree_to_CP(tree, thresholds=None):
    """Return tensor encoding tree in CP format together with thresholds.

    Parameters
    ----------
    tree : sklearn.tree.tree_.Tree

    thresholds : list[np.ndarray] (optional)
        Don't infer thresholds from tree, but use a precomputed list instead.

    Returns
    -------
    thresholds : list[np.ndarray]
        One array of thresholds for each feature. Last value is always `np.inf`

    leaf_values : np.ndarray
        Decision labels at leaves

    leaf_filter_matrices : list[np.ndarray[bool]]
        For each feature a matrix of shape (n_thresholds, n_leaves). Each row
        encodes the filter values for this particular feature and leaf.
    """

    if isinstance(tree, BaseDecisionTree):
        tree = tree.tree_

    # Compute thresholds for each feature
    feat_inds = [
        np.where(tree.feature == feat) for feat in range(tree.n_features)
    ]
    if thresholds is None:
        thresholds = []
        for inds in feat_inds:
            thresholds.append(
                np.unique(np.concatenate([[np.infty], tree.threshold[inds]]))
            )

    # Decision labels of leaves
    leaves = np.where(tree.feature < 0)
    if tree.value.shape[-1] == 2:  # Classification tree, compute logit
        leaf_values = tree.value[leaves]
        leaf_values = np.log1p(leaf_values[:, 0, 0]) - np.log1p(
            leaf_values[:, 0, 1]
        )
    else:  # Regression tree, just take values as-is
        leaf_values = tree.value[leaves][:, 0, 0]

    # For each node, initialize filter matrix to be True for all thresholds
    filter_matrices = [
        np.full((tree.node_count, len(thresholds[i])), True)
        for i in range(tree.n_features)
    ]

    for index in range(tree.node_count):
        l_child = tree.children_left[index]
        r_child = tree.children_right[index]
        if l_child < 0:  # leaf node
            continue

        feat = tree.feature[index]
        threshold = tree.threshold[index]
        new_filter = thresholds[feat] <= threshold

        # Let children inherit filter matrices
        for f in range(tree.n_features):
            filter_matrices[f][l_child] = filter_matrices[f][index].copy()
            filter_matrices[f][r_child] = filter_matrices[f][index].copy()

        # Update filter matrices of children
        filter_matrices[feat][l_child] *= new_filter
        filter_matrices[feat][r_child] *= ~new_filter

    # Only store filters of leaves
    leaf_filter_matrices = [
        filt[leaves].astype(np.float64) for filt in filter_matrices
    ]
    return thresholds, leaf_values, leaf_filter_matrices


def forest_to_CP(forest, thresholds, take_mean=True):
    """
    Convert a forest to a CP tensor.

    This function computes CP tensor for each tree, and concatenates the resulting cores.
    The leaf_values are contracted onto the first (left-most) core.

    Parameters
    ----------
    forest:
        Any `sklearn.ensemble` class using `DecisionTreeEstimator`s
    thresholds: list<np.ndarray>
        List of thresholds to use for each feature.
    take_mean: bool (default: True)
        Whether to add the trees or take the mean. Depends on which ensemble is used. For example,
        for a random forest we should average the trees, but for a boosted forest we should add
        them.
    """

    # Compute CP form of all trees
    all_leaf_values = []
    all_leaf_filter_matrices = []
    for tree in forest.estimators_:
        if isinstance(tree, np.ndarray):
            tree = tree[0]
        _, leaf_values, leaf_filter_matrices = tree_to_CP(
            tree.tree_, thresholds=thresholds
        )
        all_leaf_values.append(leaf_values)
        all_leaf_filter_matrices.append(leaf_filter_matrices)

    leaf_values = np.concatenate(all_leaf_values)
    if take_mean:
        leaf_values /= forest.n_estimators

    # concatenate all the results
    cores = []
    for i in range(len(thresholds)):
        core = np.concatenate([M[i] for M in all_leaf_filter_matrices]).T
        if i == 0:
            core *= leaf_values
        cores.append(core)

    return cores


def CP_to_TT(
    cp_cores,
    max_rank,
    eps=1e-8,
    final_round=None,
    rsvd_kwargs=None,
    verbose=False,
):
    """
    Approximate a CP tensor by a TT tensor.

    All cores of the TT are rounded to have a TT-rank of most `max_rank`, and singular values of at
    most `eps` times the largest singular value. For the first core and last core this rounding is
    done using SVD, for all other cores a randomized SVD is employed. Uses
    `sklearn.utils.extmath.randomized_svd¶` for randomized SVD. After forming the TT, it is
    optionally rounded again with an accuracy of `final_round`.

    Parameters
    ----------
    cp_cores: list<np.ndarray>
        List of CP cores
    max_rank: int
    eps: float (default: 1e-8)
    rsvd_kwargs: dict (optional)
        keyword arguments to pass to the randomized svd method.
    verbose: bool (default: False)
    """
    d = len(cp_cores)
    tt_cores = [None] * d
    prev_rank = 1

    if rsvd_kwargs is None:
        rsvd_kwargs = {}

    for alpha in range(d):
        core = cp_cores[alpha]
        dim = core.shape[0]
        if alpha == 0:
            U, S, V = svd(cp_cores[0], full_matrices=False)
        elif alpha < d - 1:  # Use randomized SVD for middle cores
            core = np.einsum("ik,jk->ijk", SV, core)
            core_mat = core.reshape(
                core.shape[0] * core.shape[1], core.shape[2]
            )
            U, S, V = randomized_svd(
                core_mat, n_components=max_rank, **rsvd_kwargs
            )
        else:  # alpha = d - 1
            core = np.einsum("ik,jk->ij", SV, core)
            U, S, V = svd(core)
            r = 1
        r = max(1, min(max_rank, np.sum(S > eps)))
        U = U[:, :r]
        S = S[:r]
        V = V[:r, :]
        SV = (S * V.T).T
        if alpha == d - 1:
            tt_cores[alpha - 1] = np.einsum(
                "ijk,kl->ijl", tt_cores[alpha - 1], U
            )
            tt_cores[alpha] = SV.reshape(SV.shape + (1,))
        else:
            tt_cores[alpha] = U.reshape((prev_rank, dim, r))
        if verbose:
            print(
                f"feature {alpha+1}/{d}, compressed TT core size is {tt_cores[alpha].shape}"
            )
        prev_rank = r
    if verbose:
        print("Orthogonalizing")
    tt = TensorTrain(tt_cores, is_orth=True)
    if final_round is not None:
        if verbose:
            print(f"Rounding to {final_round}...")
        tt.round(eps=final_round)
    if verbose:
        print(f"Final TT rank: {tt.tt_rank}")
    return tt
