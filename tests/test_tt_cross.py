import numpy as np
from numpy.core.numeric import allclose
import pytest

import sys

sys.path.insert(1, "..")
from ttml.tensor_train import TensorTrain

from ttml.tt_cross import (
    _apply_inv_matrix,
    _apply_matrix,
    _init_tt_cross,
    _maxvol_tensor,
    _qr_tensor,
    _supercore_index,
    _sweep_step_dmrg,
    index_function_wrapper,
    maxvol,
)
from ttml.tensor_train import TensorTrain


@pytest.mark.parametrize("shape", ((100, 10), (100, 50), (10, 10), (20, 10)))
def test_maxvol(shape):
    n, r = shape
    A = np.random.normal(size=shape)
    A, _ = np.linalg.qr(A)
    ind = maxvol(A)
    maxvol_det = np.linalg.det(A[ind])

    random_det = np.max(
        [
            np.abs(np.linalg.det(A[np.random.choice(n, size=r, replace=False)]))
            for _ in range(1000)
        ]
    )

    # compare to random determinant. It's possible (but unlikely) this is
    # slightly better, so multiply by small constant.
    assert random_det <= np.abs(maxvol_det)*1.2


@pytest.mark.parametrize("size", ((20, 9, 8, 2), (5, 5, 5)))
def test_inv_matrix(size):
    C = np.random.normal(size=size)
    for i, s in enumerate(size):
        A = np.random.normal(size=(s, s))
        C1 = _apply_inv_matrix(C, A, i)
        C2 = _apply_matrix(C, np.linalg.pinv(A).T, i)
        assert np.allclose(C1, C2)


@pytest.mark.parametrize("size", ((20, 9, 8, 2), (5, 5, 5)))
def test_qr_tensor(size):
    X = np.random.normal(size=size)
    for i in range(len(size)):
        Q, R = _qr_tensor(X, i)
        X2 = _apply_matrix(Q, R, i)
        assert np.allclose(X, X2)


def random_inds(shape, n, free_axis=None):
    inds = []
    for i, s in enumerate(shape):
        if free_axis is not None and i == free_axis:
            new_inds = slice(None, None)
        else:
            new_inds = np.random.choice(s, size=n)
        inds.append(new_inds)
    inds = tuple(inds)
    return inds


def random_det(X, i):
    inds = random_inds(X.shape, X.shape[i], free_axis=i)
    return np.linalg.det(X[inds])


@pytest.mark.parametrize("size", ((5, 5, 5), (10, 4, 6)))
def test_maxvol_tensor(size):
    X = np.random.normal(size=size)
    for i in range(len(size)):
        inds, R = _maxvol_tensor(X, i)
        rand_det_X = np.max([np.abs(random_det(X, i)) for _ in range(100)])
        R_det = np.abs(np.linalg.det(R))
        assert R_det >= rand_det_X

        inds = list(inds)
        inds = inds[:i] + [slice(None, None)] + inds[i:]
        R_from_inds = X[tuple(inds)]
        if i == 0:
            R_from_inds = R_from_inds.T
        assert np.allclose(R_from_inds, R)


def verify_pmat(tt, index, index_array, direction):
    if direction == "RL":
        P = np.zeros((tt.tt_rank[index - 1],) * 2)
        for j in range(index_array[index].shape[1]):
            row = None
            for i, k in enumerate(range(len(tt) - 1, index - 1, -1)):
                if row is None:
                    row = tt[k][:, index_array[index][-i - 1, j], 0]
                else:
                    row = tt[k][:, index_array[index][-i - 1, j], :] @ row
            P[:, j] = row
    if direction == "LR":
        P = np.zeros((tt.tt_rank[index - 1],) * 2)
        for j in range(index_array[index].shape[1]):
            row = None
            for i in range(index):
                if row is None:
                    row = tt[i][0, index_array[index][i, j], :]
                else:
                    row = row @ tt[i][:, index_array[index][i, j], :]
            P[:, j] = row
    return P


@pytest.mark.parametrize(
    "dims,ranks", [((4, 4, 4, 4), (2, 2, 2)), ((5, 8, 3), (5, 3))]
)
def test_init_dmrg_cross(dims, ranks):
    # This also confirms the validity of compute_multi_indices in the RL
    # direction
    tt = TensorTrain.random(dims, ranks)
    tt_old = tt.copy()
    tt, P_mats, index_array = _init_tt_cross(tt)

    assert (tt - tt_old).norm() < 1e-8
    for i in range(1, len(tt)):
        P1 = P_mats[i]
        P2 = verify_pmat(tt, i, index_array, "RL")
        assert np.allclose(P1, P2)


def big_index_simple(left_indices, right_indices, dim1, dim2, tensor_order):
    """
    Simple implementation of indices used for constructing DMRG core,
    deprecated but useful for testing.
    """
    if left_indices is not None:
        rank_left = left_indices.shape[1]
    else:
        rank_left = 1
    if right_indices is not None:
        rank_right = right_indices.shape[1]
    else:
        rank_right = 1
    big_index = np.zeros(
        (rank_left, dim1, dim2, rank_right, tensor_order), dtype=int
    )
    if (left_indices is not None) and (right_indices is not None):
        for i1 in range(rank_left):
            for i2 in range(rank_right):
                for s1 in range(dim1):
                    for s2 in range(dim2):
                        big_index[i1, s1, s2, i2, :] = np.concatenate(
                            [
                                left_indices[:, i1],
                                [s1],
                                [s2],
                                right_indices[:, i2],
                            ]
                        )
    elif (left_indices is None) and (right_indices is not None):
        for i2 in range(rank_right):
            for s1 in range(dim1):
                for s2 in range(dim2):
                    big_index[0, s1, s2, i2, :] = np.concatenate(
                        [[s1], [s2], right_indices[:, i2]]
                    )
    elif (left_indices is not None) and (right_indices is None):
        for i1 in range(rank_left):
            for s1 in range(dim1):
                for s2 in range(dim2):
                    big_index[i1, s1, s2, 0, :] = np.concatenate(
                        [left_indices[:, i1], [s1], [s2]]
                    )
    else:  # left and right inds, are none; this is matrix case
        for s1 in range(dim1):
            for s2 in range(dim2):
                big_index[0, s1, s2, 0, :] = np.concatenate([[s1], [s2]])
    return big_index


def check_big_ind(i, index_array, tt):
    left_inds = index_array[i]
    right_inds = index_array[i + 2]
    big_ind1 = big_index_simple(
        left_inds, right_inds, tt.dims[i], tt.dims[i + 1], len(tt)
    )
    big_ind2 = _supercore_index(
        left_inds, right_inds, tt.dims[i], tt.dims[i + 1]
    )
    return np.allclose(big_ind1, big_ind2)


@pytest.mark.parametrize(
    "dims,ranks", [((10, 10, 10, 10), (5, 8, 6)), ((5, 5, 5, 5), (5, 25, 5))]
)
def test_sweep_step(dims, ranks):
    tt = TensorTrain.random(dims, ranks)
    tt_target = TensorTrain.random(dims, ranks)
    index_fun = index_function_wrapper(tt_target.gather)

    tt, Pmats, index_array = _init_tt_cross(tt)

    for _ in range(2):
        for i in range(len(tt) - 1):
            _sweep_step_dmrg(i, "LR", tt, index_array, index_fun, Pmats, rank_kick=0)
            pmat_check = np.allclose(
                verify_pmat(tt, i + 1, index_array, "LR").T, Pmats[i + 1]
            )
            assert pmat_check
            assert check_big_ind(i, index_array, tt)

        for i in range(2, -1, -1):
            _sweep_step_dmrg(i, "RL", tt, index_array, index_fun, Pmats, rank_kick=0)
            pmat_check = np.allclose(
                verify_pmat(tt, i + 1, index_array, "RL"), Pmats[i + 1]
            )
            assert pmat_check
            assert check_big_ind(i, index_array, tt)

        error = (tt - tt_target).norm()
    assert error < 1e-8
