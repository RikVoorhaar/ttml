"""
Implementation of (DMRG) TT-cross algorithm like in the Matlab `TT-Toolbox
<https://github.com/oseledets/TT-Toolbox>`_. 

The (DMRG) TT-cross algorithm is an efficient method to approximate black-box
functions with tensor trains. For this library, its main usage is to initialize
the Tensor Train used in the TTML estimator. The algorithm can however
be used for many other purposes as well.

For example, below we use TT-DMRG cross to approximate the sine function in
5 dimensions:

>>> from ttml.tt_cross import estimator_to_tt_cross
... import numpy as np
...
... def f(x):
...     return np.sin(np.sum(x, axis=1))
...
... # Same thresholds for every feature: [0,0.2,0.4,0.6,0.8]
... thresholds = [np.linspace(0, 1, 11)] * 5
...
... tt = estimator_to_tt_cross(f, thresholds)
... tt.gather(np.array([[2, 2, 2, 2, 2]])) - np.sin(1)
array([3.33066907e-16])

Note here that the index ``(2, 2, 2, 2, 2)`` corresponds to the point
``(0.2, 0.2, 0.2, 0.2, 0.2)``. It turns out that this function admits a low-rank
tensor train decomposition. In fact, we can look at the singular values of this
tensor train:

>>> tt.sing_vals()
[array([2.62706082e+02, 1.10057236e+02, 4.55274516e-14, 3.68166789e-14, 3.26467439e-14]),
 array([2.50489642e+02, 1.35580309e+02, 4.13492807e-14, 1.73287140e-14, 6.94636648e-15]),
 array([2.50489642e+02, 1.35580309e+02, 1.39498787e-14, 1.02576584e-14, 6.98685871e-15]),
 array([2.62706082e+02, 1.10057236e+02, 2.02285923e-14, 1.07720967e-14, 9.56667996e-15])]

We see that effectively the tensor train has rank 2. We can control the rank
by setting the `max_rank` keyword argument in :meth:`estimator_to_tt_cross`:

>>> tt2 = estimator_to_tt_cross(f, thresholds, max_rank=2)
... tt2
<TensorTrain of order 5 with outer dimensions (11, 11, 11, 11, 11), 
TT-rank (2, 2, 2, 2), and orthogonalized at mode 4>

And indeed, ``tt2`` is very close to ``tt``:

>>> (tt - tt2).norm()
4.731228127074942e-13

We can do this for any function, and indeed for ``TTML`` we use a machine
learning estimator instead of the function ``f`` above. For example we can use
this to obtain a tensor train approximating a random forest's :meth:`.predict`
method.

>>> from sklearn.ensemble import RandomForestRegressor
...
... X = np.random.normal(size=(1000, 5))
... y = np.exp(np.sum(X, axis=1))
...
... forest = RandomForestRegressor()
... forest.fit(X, y)
...
... thresholds = [np.linspace(0, 1, 11)] * 5
...
... tt = estimator_to_tt_cross(forest.predict, thresholds, max_rank=2)
... tt
<TensorTrain of order 5 with outer dimensions (11, 11, 11, 11, 11),
TT-rank (2, 2, 2, 2), and orthogonalized at mode 4>


We implemented two versions of the TT-cross approximation algorithm. A `'dmrg'`
and a `'regular'` version. The default is the `'dmrg'` version, and it optimizes
the TT one core at a time in alternating left-to-right and right-to-left sweeps.
The `'dmrg'` version optimizes two cores at the same time. The latter approach
is more costly numerically speaking, but has the potential ability to estimate
the rank of the underlying TT automatically (although this is not an implemented
feature). The DMRG algorithm also converges faster, and tends to result in a
better final test error. Therefore the DMRG is the default, despite the fact
that it is slower. We can control which version is used through the `method`
kwarg:

>>> tt = estimator_to_tt_cross(forest.predict, thresholds, method='dmrg')
"""

import numpy as np
import scipy.linalg
import scipy.linalg.lapack
from copy import copy

from ttml.tensor_train import TensorTrain
from ttml.utils import random_idx


def _piv_to_ind(piv, n):
    """Convert pivots returned by LAPACK to permutation indices"""
    ind = np.arange(n)
    for i, j in enumerate(piv):
        temp = ind[i]
        ind[i] = ind[j]
        ind[j] = temp
    return ind


def _right_solve(A, B):
    """Solve the linear equation XA =B

    can be rephrased as A' X' = B'"""
    return scipy.linalg.solve(A.T, B.T).T


def maxvol(A, eps=1e-2, niters=100):
    """
    Quasi-max volume submatrix

    Initializes with pivot indices of LU decomposition, then greedily
    interchanges rows.
    """
    n, r = A.shape
    if n <= r:
        return np.arange(n)
    A, _ = scipy.linalg.qr(A, mode="economic")
    out = scipy.linalg.lapack.dgetrf(A)  # LU decomp

    _, P, _ = out
    ind = _piv_to_ind(P, n)[:r]

    sbm = A[ind[:r]]
    b = _right_solve(sbm, A)

    for _ in range(niters):
        i0, j0 = np.unravel_index(np.argmax(np.abs(b)), b.shape)
        mx0 = b[i0, j0]
        if np.abs(mx0) <= 1 + eps:
            break
        k = ind[j0]
        b += np.outer(b[:, j0], b[k, :] - b[i0, :]) / mx0
        ind[j0] = i0
    ind.sort()
    return ind


def _apply_inv_matrix(X, A, mu):
    """Apply the (pseudo)inverse of `A` to tensor `X` at mode `mu`.

    First transpose and reshape X so that its matrix with first mode the tensor
    of interest. Then solve linear system, and undo the reshape and tranpose.
    """

    if A.shape[0] != X.shape[mu]:
        raise ValueError(
            f"Incompatible shapes, {A.shape} and {X.shape} at mode {mu}"
        )

    # Flip modes mu and 0
    permutation = np.arange(len(X.shape))
    permutation[mu] = 0
    permutation[0] = mu

    Y = X.transpose(permutation)
    Y_shape = Y.shape
    Y = Y.reshape(A.shape[0], -1)
    if A.shape[0] == A.shape[1]:
        Y = np.linalg.solve(A, Y)
    else:
        Y = np.linalg.lstsq(A, Y, rcond=None)[0]
    Y = Y.reshape((A.shape[1],) + Y_shape[1:])
    Y = Y.transpose(permutation)
    return Y


def _qr_tensor(X, mu, transpose=False):
    """Orthogonalize tensor with respect to specified mode

    Output is tensor Q of the same shape (unless there are rank deficiencies),
    and a matrix R such that `np.einsum("...i...,ij->...j...", Q, R) = X`.

    This is done by matricizing with respect to mode `mu`, then taking QR, and
    undoing the matricization ofQ"""
    permutation = np.arange(len(X.shape))
    permutation[mu] = permutation[-1]
    permutation[-1] = mu
    Y = X.transpose(permutation)
    Y_shape = Y.shape
    Y = Y.reshape(-1, X.shape[mu])
    Q, R = scipy.linalg.qr(Y, mode="economic")
    Q = Q.reshape(Y_shape[:-1] + (Q.shape[1],))
    Q = Q.transpose(permutation)
    if transpose:
        R = R.T
    return Q, R


def _apply_matrix(X, A, mu):
    """Apply a matrix A to X at a specific mode, summing over first index."""
    # use einsum string for X of form "abcd..."
    X_inds = "".join([chr(ord("a") + i) for i in range(len(X.shape))])

    # for A, take mu'th letter of alphabet + a new letter
    A_inds = X_inds[mu] + chr(ord("a") + len(X.shape))

    # in output einsum string replace mu'th letter by the new letter
    out_inds = list(X_inds)
    out_inds[mu] = A_inds[-1]
    out_inds = "".join(out_inds)
    ein_string = f"{X_inds},{A_inds}->{out_inds}"

    # Do the summation and return
    return np.einsum(ein_string, X, A)


def _maxvol_tensor(X, mu, transpose=False):
    """
    Matricize `X` with respect to mode `mu` and return maxvol submatrix and
    indices
    """
    permutation = np.concatenate(
        [np.arange(mu), np.arange(mu + 1, len(X.shape)), [mu]]
    )
    Y = X.transpose(permutation)
    Y = Y.reshape(-1, X.shape[mu])
    ind = maxvol(Y)
    R = Y[ind, :]
    ind = np.unravel_index(ind, X.shape[:mu] + X.shape[mu + 1 :])
    ind = np.stack(ind)
    if transpose:
        R = R.T
    return ind, R


def _compute_multi_indices(ind, ind_old, direction):
    """
    Compute new multiindex from old multiindex and pairs of (alpha_{k-1},i_k) as
    described in Savistyanov-Oseledets. This guarantees a nested sequence of
    multiindices, and works for both the left and right indices.
    """
    r = ind.shape[1]
    if direction == "RL":
        dim_indices, previous_indices = ind
    else:
        previous_indices, dim_indices = ind

    if ind_old is None:
        return dim_indices.reshape(1, r)
    else:
        ind_new = np.zeros((len(ind_old) + 1, r), dtype=np.int32)
        if direction == "RL":
            ind_new[1:, :] = ind_old[:, previous_indices]
            ind_new[0, :] = dim_indices
        elif direction == "LR":
            ind_new[:-1, :] = ind_old[:, previous_indices]
            ind_new[-1, :] = dim_indices
        else:
            raise ValueError("Direction has to be 'LR' or 'RL'")
    return ind_new


def _init_tt_cross(tt):
    """Generate initial set of R-matrices and right-indices for TT-cross.

    Same for DMRG as for regular TT-cross. This version follows the paper
    instead of Matlab code."""
    tt.orthogonalize("l")
    nd = len(tt)
    P_mats = [None] * (nd + 1)
    P_mats[0] = np.array([[1]])
    P_mats[-1] = np.array([[1]])
    index_array = [None] * (nd + 1)
    R = np.array([[1]])

    for i in range(nd - 1, 0, -1):
        core = tt[i]
        core = np.einsum("ijk,kl->ijl", core, R)

        # RQ decomposition of core
        Q, R = _qr_tensor(core, 0, True)

        tt[i] = Q
        Q = np.einsum("ijk,kl", Q, P_mats[i + 1])

        # Max vol indices
        # ind = maxvol(core.T)
        ind, P = _maxvol_tensor(Q, 0, True)
        P_mats[i] = P

        # Compute new indices from previous and maxvol
        ind_new = _compute_multi_indices(ind, index_array[i + 1], "RL")
        index_array[i] = ind_new
    tt[0] = np.einsum("ijk,kl->ijl", tt[0], R)

    # tt is now right-orthogonalized and identical to what we started with
    tt.mode = 0
    return tt, P_mats, index_array


def _supercore_index(left_indices, right_indices, dim1, dim2):
    """
    Return indices corresponding to the DMRG cross defined by fixing
    `left_indices` and `right_indices`, and with dimensions of the two
    respective indices given by `dim1` and `dim2`.
    """
    # If left_indices is None, then there we are at the very left of the train.
    # In this case we treat the rank to the left as 1.
    if left_indices is None:
        rank1 = 1
    else:
        rank1 = left_indices.shape[1]

    # Very right of the train
    if right_indices is None:
        rank2 = 1
    else:
        rank2 = right_indices.shape[1]

    # Creating arrays of shape (rank1, dim1, dim2, rank2) for each index
    i1, s1, s2, i2 = np.meshgrid(
        np.arange(rank1),
        np.arange(dim1),
        np.arange(dim2),
        np.arange(rank2),
        indexing="ij",
    )
    shape = i1.shape  # all shapes are equal
    to_concatenate = []

    # Use i1 to sample left_indices
    if left_indices is not None:
        left = left_indices[:, i1.reshape(-1)].T.reshape(
            shape + (left_indices.shape[0],)
        )
        to_concatenate.append(left)
    # concatenate s1 and s2 to the left_indices
    to_concatenate.append(s1.reshape(s1.shape + (1,)))
    to_concatenate.append(s2.reshape(s2.shape + (1,)))

    # finally concatenate right_indices to the lists
    if right_indices is not None:
        right = right_indices[:, i2.reshape(-1)].T.reshape(
            shape + (right_indices.shape[0],)
        )
        to_concatenate.append(right)

    big_ind = np.concatenate(to_concatenate, axis=-1)
    return big_ind


def _core_index(left_indices, right_indices, dim):
    """
    Return indices corresponding to the cross defined by fixing
    `left_indices` and `right_indices`, and with dimension of the index
    respective indices given by `dim`.
    """
    # If left_indices is None, then there we are at the very left of the train.
    # In this case we treat the rank to the left as 1.
    if left_indices is None:
        rank1 = 1
    else:
        rank1 = left_indices.shape[1]

    # Very right of the train
    if right_indices is None:
        rank2 = 1
    else:
        rank2 = right_indices.shape[1]

    # Creating arrays of shape (rank1, dim, rank2) for each index
    i1, s1, i2 = np.meshgrid(
        np.arange(rank1),
        np.arange(dim),
        np.arange(rank2),
        indexing="ij",
    )
    shape = i1.shape  # all shapes are equal
    to_concatenate = []

    # Use i1 to sample left_indices
    if left_indices is not None:
        left = left_indices[:, i1.reshape(-1)].T.reshape(
            shape + (left_indices.shape[0],)
        )
        to_concatenate.append(left)
    # concatenate s1 and s2 to the left_indices
    to_concatenate.append(s1.reshape(s1.shape + (1,)))

    # finally concatenate right_indices to the lists
    if right_indices is not None:
        right = right_indices[:, i2.reshape(-1)].T.reshape(
            shape + (right_indices.shape[0],)
        )
        to_concatenate.append(right)

    big_ind = np.concatenate(to_concatenate, axis=-1)
    return big_ind


def index_function_wrapper(fun):
    """
    Modify a multi-index function to accept multi-dimensional arrays of
    multi-indices.
    """

    def new_fun(inds):
        shape = inds.shape
        new_inds = inds.reshape(-1, shape[-1])
        out = fun(new_inds)
        return out.reshape(shape[:-1])

    return new_fun


# TODO: This is really slow. Probably best would be to do this in Cython.
def index_function_wrapper_with_cache(fun):
    """
    Modify a multi-index function to accept multi-dimensional arrays of
    multi-indices.
    """
    cache = dict()

    def new_fun(inds):
        shape = inds.shape
        flat_inds = inds.reshape(-1, shape[-1])
        result = np.ones(len(flat_inds))
        for i, row in enumerate(flat_inds):
            result[i] = cache.get(tuple(row), np.nan)
        new_inds = np.nonzero(np.isnan(result))[0]
        if len(new_inds) > 0:
            out = fun(flat_inds[new_inds])
            # print(out.shape, flat_inds[new_inds].shape, new_inds.shape,flat_inds.shape)
            result[new_inds] = out
            for row, res in zip(flat_inds[new_inds], out):
                cache[tuple(row)] = res

        return result.reshape(shape[:-1])

    return new_fun


def _sweep_step_dmrg(
    i,
    direction,
    tt,
    index_array,
    index_fun,
    Pmats,
    rank_kick=0,
    verbose=False,
    cache=None,
):
    """
    Do one step of the DMRG TT-cross algorithm, sweeping in a specified
    direction.

    Parameters
    ----------
    i : int
        Left index of the supercore
    direction : str
        Either "LR" or "RL", corresponding to a sweep in the left-to-right
        and right-to-left direcition respectively
    tt : TensorTrain
        TensorTrain to be modified
    index_array : list[np.ndarray]
        list of left_indices and right_indices. At step `i`, `index_array[i+1]`
        will be modified
    index_fun : function
        Function mapping indices to function values to be used for fitting
    Pmats : list[np.ndarray]
        The list of matrices to be used to compute maxvol at step i. At step
        `i`, `Pmats[i+1]` will be modified.
    rank_kick : int (default: 0)
        Increase rank each step by this amount (if possible)
    verbose: bool (default: False)
        Print convergence information every step.
    """
    ranks = (1,) + tt.tt_rank + (1,)
    dims = tt.dims

    # Compute indices used to sample supercore
    left_inds = index_array[i]
    right_inds = index_array[i + 2]
    big_ind = _supercore_index(
        left_inds, right_inds, tt.dims[i], tt.dims[i + 1]
    )

    # Construct supercore
    super_core = index_fun(big_ind)
    if cache is not None:
        cache["func_vals"] = np.concatenate(
            [cache["func_vals"], super_core.reshape(-1)]
        )
        cache["inds"] = np.concatenate(
            [cache["inds"], big_ind.reshape(-1, len(tt))]
        )

    # Multiply supercore by (pseudo)inverse of the P-matrices on the left and
    # right edge.
    super_core = _apply_inv_matrix(super_core, Pmats[i], 0)
    super_core = _apply_inv_matrix(super_core, Pmats[i + 2].T, 3)

    # Compute the 'old' to compare and compute local convergence
    old_super_core = np.einsum("iaj,jbk->iabk", tt[i], tt[i + 1])
    local_error = np.linalg.norm(super_core - old_super_core) / np.linalg.norm(
        super_core
    )
    if verbose:
        print(f"local error at {i=}, {direction=}: {local_error=:.4e}")

    # TODO: use fast truncated SVD
    # Split supercore using SVD
    U, S, V = scipy.linalg.svd(
        super_core.reshape(ranks[i] * dims[i], dims[i + 1] * ranks[i + 2]),
        full_matrices=False,
    )
    r = min(
        ranks[i + 1] + rank_kick, len(S)
    )  # fixed rank for now, make better later
    U = U[:, :r]
    S = S[:r]
    V = V[:r, :]

    # If direction is LR, absorb singular values into right core (V), otherwise
    # we absorb them into the left core (U)
    if direction == "LR":
        V = np.einsum("i,ij->ij", S, V)
    else:  # direction is RL
        U = np.einsum("ij,j->ij", U, S)

    # Reshape left and right core into tensors and store them
    U = U.reshape(ranks[i], dims[i], r)
    V = V.reshape(r, dims[i + 1], ranks[i + 2])
    tt[i] = U
    tt[i + 1] = V

    # Store new rank into TT, in case it changed
    ttr = list(tt.tt_rank)
    ttr[i] = r
    tt.tt_rank = tuple(ttr)

    # Use maxvol to compute new P matrix and nested multi-indices
    if direction == "LR":
        U = np.einsum("ij,jkl->ikl", Pmats[i], U)
        ind, P = _maxvol_tensor(U, 2)
        Pmats[i + 1] = P
        new_indices = _compute_multi_indices(ind, left_inds, "LR")
        index_array[i + 1] = new_indices

    if direction == "RL":
        V = np.einsum("ijk,kl", V, Pmats[i + 2])
        ind, P = _maxvol_tensor(V, 0, True)
        Pmats[i + 1] = P
        new_indices = _compute_multi_indices(ind, right_inds, "RL")
        index_array[i + 1] = new_indices

    return local_error


def tt_cross_dmrg(
    tt, index_fun, tol=1e-3, max_its=10, verbose=False, inplace=True
):
    """
    Implements DMRG TT-Cross algorithm

    Recovers a tensor-train from a function mapping indices to numbers. The
    function `index_fun` should accept arbitrary multidimensional arrays of
    indices, with last axis the same shape as the number of dimensions. You can
    use `index_function_wrapper` to convert a function to this form.

    Parameters
    ----------
    tt: TensorTrain
    index_fun: function
    tol: float (default: 1e-3)
        Tolerance for convergence. The algorithm is stopped if after a half-
        sweep the maximum difference in the half-sweep between any cross-sampled
        supercore and supercore of the TT is less than `tol`.
    max_its: int (default: 5)
    verbose: bool (default: False)
    inplace: bool (default: True)

    Returns
    -------
    tt: TensorTrain
    """
    if not inplace:
        tt = tt.copy()
    tt, Pmats, index_array = _init_tt_cross(tt)

    errors = []
    direction = "LR"
    for j in range(max_its):
        max_local_error = -np.inf
        if direction == "LR":  # Left-right sweep
            for i in range(len(tt) - 1):
                local_error = _sweep_step_dmrg(
                    i, "LR", tt, index_array, index_fun, Pmats
                )
                max_local_error = np.max([max_local_error, local_error])
            direction = "RL"
        else:  # Right-left sweep
            for i in range(len(tt) - 2, -1, -1):
                local_error = _sweep_step_dmrg(
                    i, "RL", tt, index_array, index_fun, Pmats
                )
                max_local_error = np.max([max_local_error, local_error])
            direction = "LR"
        if verbose:
            print(
                f"Sweep {j}, direction {direction[::-1]}. {max_local_error=:.4e}"
            )
        if max_local_error < tol:
            break
    tt.orthogonalize()
    tt.errors = errors
    return tt


def _sweep_step_regular(
    i, direction, tt, index_array, index_fun, Pmats, cache=None, verbose=False
):
    """
    Do one step of the DMRG TT-cross algorithm, sweeping in a specified
    direction.

    Parameters
    ----------
    i : int
        Left index of the supercore
    direction : str
        Either "LR" or "RL", corresponding to a sweep in the left-to-right
        and right-to-left direcition respectively
    tt : TensorTrain
        TensorTrain to be modified
    index_array : list[np.ndarray]
        list of left_indices and right_indices. At step `i`, `index_array[i+1]`
        will be modified
    index_fun : function
        Function mapping indices to function values to be used for fitting
    Pmats : list[np.ndarray]
        The list of matrices to be used to compute maxvol at step i. At step
        `i`, `Pmats[i+1]` will be modified.
    verbose: bool (default: False)
        Print convergence information every step.
    """

    # Compute indices used to sample supercore
    left_inds = index_array[i]
    right_inds = index_array[i + 1]
    big_ind = _core_index(left_inds, right_inds, tt.dims[i])

    core = index_fun(big_ind)
    if cache is not None:
        cache["func_vals"] = np.concatenate(
            [cache["func_vals"], core.reshape(-1)]
        )
        cache["inds"] = np.concatenate(
            [cache["inds"], big_ind.reshape(-1, len(tt))]
        )

    # Multiply core by (pseudo)inverse of the P-matrices on the left and
    # right edge.
    core = _apply_inv_matrix(core, Pmats[i], 0)
    core = _apply_inv_matrix(core, Pmats[i + 1].T, 2)

    # Orthogonalize and absorb R-factor into next core
    if direction == "LR" and (i < len(tt) - 1):
        core, R = _qr_tensor(core, 2, False)
        tt[i + 1] = np.einsum("ij,jkl->ikl", R, tt[i + 1])
    elif direction == "RL" and (i > 0):
        core, R = _qr_tensor(core, 0, True)
        tt[i - 1] = np.einsum("ijk,kl->ijl", tt[i - 1], R)

    tt[i] = core

    # Use maxvol to compute new P matrix and nested multi-indices
    if direction == "LR" and i < len(tt) - 1:
        A = np.einsum("ij,jkl->ikl", Pmats[i], core)
        ind, P = _maxvol_tensor(A, 2)
        Pmats[i + 1] = P
        new_indices = _compute_multi_indices(ind, left_inds, "LR")
        index_array[i + 1] = new_indices

    if direction == "RL" and i > 0:
        A = np.einsum("ijk,kl", core, Pmats[i + 1])
        ind, P = _maxvol_tensor(A, 0, True)
        Pmats[i] = P
        new_indices = _compute_multi_indices(ind, right_inds, "RL")
        index_array[i] = new_indices


def _test_accuracy_tt(tt, cache, max_num_samples=1000):
    "Test MSE regression error on samples to measure convergence"
    N = len(cache["func_vals"])
    num_samples = min(max_num_samples, N)
    subset = np.random.choice(N, size=num_samples, replace=False)
    y_pred = tt.gather(cache["inds"][subset])
    y_true = cache["func_vals"][subset]
    return np.mean((y_pred - y_true) ** 2)


def tt_cross_regular(
    tt, index_fun, tol=1e-2, max_its=10, verbose=False, inplace=True
):
    """
    Implements DMRG TT-Cross algorithm

    Recovers a tensor-train from a function mapping indices to numbers. The
    function `index_fun` should accept arbitrary multidimensional arrays of
    indices, with last axis the same shape as the number of dimensions. You can
    use `index_function_wrapper` to convert a function to this form.

    Parameters
    ----------
    tt: TensorTrain
    index_fun: function
    tol: float (default: 1e-8)
        Tolerance for convergence. The algorithm is stopped if after a half-
        sweep the maximum difference in the half-sweep between any cross-sampled
        supercore and supercore of the TT is less than `tol`.
    max_its: int (default: 5)
    verbose: bool (default: False)
    inplace: bool (default: True)

    Returns
    -------
    tt: TensorTrain
    """
    if not inplace:
        tt = tt.copy()
    tt, Pmats, index_array = _init_tt_cross(tt)
    direction = "LR"
    if tol is not None:
        cache = dict()
        cache["inds"] = random_idx(tt, 200)
        cache["func_vals"] = index_fun(cache["inds"]).reshape(-1)

    errors = []
    for j in range(max_its):
        if direction == "LR":  # Left-right sweep
            for i in range(len(tt)):
                _sweep_step_regular(
                    i,
                    "LR",
                    tt,
                    index_array,
                    index_fun,
                    Pmats,  # cache=cache
                )
            if tol is not None:
                errors.append(_test_accuracy_tt(tt, cache))
            direction = "RL"
        else:  # Right-left sweep
            for i in range(len(tt) - 1, -1, -1):
                _sweep_step_regular(
                    i,
                    "RL",
                    tt,
                    index_array,
                    index_fun,
                    Pmats,  # cache=cache
                )
            if tol is not None:
                errors.append(_test_accuracy_tt(tt, cache))
            direction = "LR"
        if verbose:
            print(f"Sweep {j}, direction {direction[::-1]}. ")
            if tol is not None:
                print(f"Last error: {errors[-1]:.4e}")
        # check for convergence, and stop if converged
        if tol is not None and len(errors) > 3:
            if errors[-1] > (1 - tol) * (np.max(errors[-4:-1])):
                break

    tt.orthogonalize()
    tt.errors = np.array(errors)
    return tt


def _inds_to_X(thresh, inds):
    """Use thresholds to turn indices into data values"""
    X = [t[i] for t, i in zip(thresh, inds)]
    X = np.stack(X, axis=-1)
    return X


def _thresh_to_index_fun(thresh, predict_method, use_cache=False):
    """Uses thresholds and and estimators `.predict` function to create an
    index function to be used by TT-cross.

    For a regression estimator one can use .predict, but for
    classification it is necessary to use the prediction method that
    outputs logits."""

    def fun(inds):
        X = _inds_to_X(thresh, inds.T)
        return predict_method(X)

    if use_cache:
        index_fun = index_function_wrapper_with_cache(fun)
    else:
        index_fun = index_function_wrapper(fun)

    return index_fun


def estimator_to_tt_cross(
    predict_method,
    thresholds,
    max_rank=5,
    tol=1e-2,
    max_its=5,
    method="regular",
    use_cache=False,
    verbose=False,
):
    """Use TT-cross to convert an estimator into a TT

    Parameters
    ----------
    predict_method : function
        function mapping data X onto truth labels y used for training
    thresholds : list[np.ndarray]
        List of thresholds to use for each feature. Should be a list of arrays,
        one array per feature. The last element of each array is expected to be
        `np.inf`.
    max_rank : int (default: 5)
        Maximum rank for the tensor train
    tol : float (default: 1e-8)
        Tolerance for checking convergence for the DMRG algorithm. If maximum
        local error in an entire sweep is smaller than `tol`, we stop early.
    max_its : int (default: 10)
        Number of (half) sweeps to perform
    method: : str (default: "dmrg")
        Whether to use "regular" or "dmrg" tt-cross algorithm
    use_cache : bool (default: False)
        Whether to cache function calls (Experimental, needs better
        implementation)
    verbose: bool (default: False)
        If True, print convergence information after every half sweep.
    """
    thresh_no_inf = [copy(t) for t in thresholds]
    for t in thresh_no_inf:
        t[-1] = t[-2] + 1  # Remove np.inf from thresholds

        min_gap = np.min(np.diff(t))
        t = t - min_gap / 2  # shift all thresholds away from boundary

    # Function maping indices to data values
    index_fun = _thresh_to_index_fun(
        thresh_no_inf, predict_method, use_cache=use_cache
    )

    # Init TT
    dims = tuple(len(t) for t in thresholds)
    tt = TensorTrain.random(dims, max_rank, mode="r")

    # Use TT-cross to fit
    if method == "dmrg":
        tt_cross_dmrg(
            tt,
            index_fun,
            tol=tol,
            max_its=max_its,
            verbose=verbose,
            inplace=True,
        )
    elif method == "regular":
        tt_cross_regular(
            tt,
            index_fun,
            tol=tol,
            max_its=max_its,
            verbose=verbose,
            inplace=True,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    return tt

