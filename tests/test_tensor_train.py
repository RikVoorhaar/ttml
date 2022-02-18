import autoray as ar
import numpy as np
import opt_einsum
import pytest

from ttml.tensor_train import (
    TensorTrain,
    TensorTrainTangentVector,
    contract_cores,
)
from ttml.utils import (
    SUPPORTED_BACKENDS,
    convert_backend,
    merge_sum,
    random_idx,
    random_normal,
)


def check_orthog(cores, mu):
    """Check if the TT defined by cores is orthogonalized at mode mu"""
    if mu == "l":
        mu = len(cores) - 1
    elif mu == "r":
        mu = 0
    max_err = 0
    for i, C in enumerate(cores):
        C = ar.to_numpy(C)
        shape = C.shape
        if i < mu:
            U = C.reshape(shape[0] * shape[1], shape[2])
            prod = U.T @ U
            err = np.linalg.norm(prod - np.eye(len(prod)))
            max_err = max(max_err, err)
        if i > mu:
            U = C.reshape(shape[0], shape[1] * shape[2])
            prod = U @ U.T
            err = np.linalg.norm(prod - np.eye(len(prod)))
            max_err = max(max_err, err)
    return max_err


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("force_rank", (True, False))
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_orthog(backend, r, d, order, force_rank):
    """Orthogonalize with respect to all modes. check it's correctly orthogonal, and that tensor
    didn't change"""
    tt = TensorTrain.random([d] * order, r, backend=backend, auto_rank=False)
    dense_before = tt.dense()
    norm = tt.norm()
    modes = list(range(tt.order)) + ["l", "r"]
    for mu in modes:
        dense_after0 = tt.orthogonalize(
            mode=mu, inplace=False, force_rank=force_rank
        ).dense()
        tt.orthogonalize(mode=mu, inplace=True, force_rank=force_rank)
        dense_after1 = tt.dense()
        error0 = np.linalg.norm(ar.to_numpy(dense_before - dense_after0)) / norm
        error1 = np.linalg.norm(ar.to_numpy(dense_before - dense_after1)) / norm
        assert error0 < 1e-5
        assert error1 < 1e-5
        assert check_orthog(tt.cores, mu) < 1e-5


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_round(backend, r, d, order):
    """Test round function with `eps=0`"""
    tt = TensorTrain.random([d] * order, r, backend=backend)
    tt_copy = tt.copy(deep=True)
    tt_copy.convert_backend("numpy")
    dense_before = tt_copy.dense()
    tt.round(eps=0)
    tt.convert_backend("numpy")
    dense_after = tt.dense()
    error = np.linalg.norm(dense_before - dense_after)
    assert error < 1e-5

    tt.round(max_rank=r - 1)
    for rank in tt.tt_rank:
        assert rank <= r - 1

    tt.round(eps=1e16)
    for rank in tt.tt_rank:
        assert rank == 1


def contract_tt_LR(tt):
    res_list = [np.reshape(tt.cores[0], tt.cores[0].shape[1:])]
    for C in tt.cores[1:]:
        new_res = opt_einsum.contract("ij,jkl->ikl", res_list[-1], C)
        new_res = np.reshape(
            new_res, (new_res.shape[0] * new_res.shape[1], new_res.shape[2])
        )
        res_list.append(new_res)
    return res_list


def contract_tt_RL(tt):
    res_list = [np.reshape(tt.cores[-1], tt.cores[-1].shape[:-1])]
    for C in tt.cores[-2::-1]:
        new_res = opt_einsum.contract("ijk,kl", C, res_list[-1])
        new_res = np.reshape(
            new_res, (new_res.shape[0], new_res.shape[1] * new_res.shape[2])
        )
        res_list.append(new_res)
    return res_list


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r1", [2, 4])
@pytest.mark.parametrize("r2", [3, 4])
@pytest.mark.parametrize("d", [2, 4, 6, 10])
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_contract_cores(backend, r1, r2, d, order):
    tt1 = TensorTrain.random([d] * order, r1, backend=backend)
    tt2 = TensorTrain.random([d] * order, r2, backend=backend)
    results = contract_cores(tt1.cores, tt2.cores, dir="LR", store_parts=True)
    results = [ar.to_numpy(C) for C in results]
    top_part = contract_tt_LR(tt1)
    bot_part = contract_tt_LR(tt2)
    for M1, M2, R in zip(top_part, bot_part, results):
        assert np.linalg.norm(M1.T @ M2 - R) < 1e-6

    results = contract_cores(tt1.cores, tt2.cores, dir="RL", store_parts=True)
    results = [ar.to_numpy(C) for C in results]
    top_part = contract_tt_RL(tt1)
    bot_part = contract_tt_RL(tt2)
    for M1, M2, R in zip(top_part, bot_part, results[::-1]):
        assert np.linalg.norm(M1 @ M2.T - R) < 1e-6


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("d", [2, 4, 10, 15])
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_gather(backend, r, d, order):
    tt = TensorTrain.random([d] * order, r, backend=backend)
    dense = ar.to_numpy(tt.dense())
    idx = random_idx(tt, 100, backend=backend)
    dense_gather = dense[tuple(ar.do("transpose", idx))]
    tt_gather = ar.to_numpy(tt.gather(idx))
    assert (
        np.linalg.norm(dense_gather - tt_gather) / np.linalg.norm(tt_gather)
        < 1e-6
    )


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize("d", [2, 4, 10])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_idx_gather(backend, r, d, order):
    tt = TensorTrain.random([d] * order, r, backend=backend)
    idx = random_idx(tt, 100, backend=backend)
    for alpha in range(len(tt)):
        env = tt.idx_env(alpha, idx, flatten=False)
        core = ar.do("take", tt[alpha], idx[:, alpha], axis=1)
        error = ar.do(
            "linalg.norm",
            opt_einsum.contract("ijk,ijk->j", env, core) - tt.gather(idx),
        )
        assert error < 1e-6
    if order > 2:
        for alpha in range(len(tt) - 1):
            env = tt.idx_env(alpha, idx, num_cores=2, flatten=False)
            env_flat = tt.idx_env(alpha, idx, num_cores=2, flatten=True)
            assert (
                np.abs(
                    ar.do("linalg.norm", env) - ar.do("linalg.norm", env_flat)
                )
                < 1e-8
            )
            core1 = ar.do("take", tt[alpha], idx[:, alpha], axis=1)
            core2 = ar.do("take", tt[alpha + 1], idx[:, alpha + 1], axis=1)
            core = opt_einsum.contract("ijk,kjl->ijl", core1, core2)
            error = ar.do(
                "linalg.norm",
                opt_einsum.contract("ijk,ijk->j", env, core) - tt.gather(idx),
            )
            assert error < 1e-6


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize("d", [2, 4, 10])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_tt_arithmetic(backend, r, d, order):
    tt0 = TensorTrain.random([d] * order, r, backend=backend)
    tt1 = tt0 + tt0
    tt1 += tt0
    assert np.abs(tt1 @ tt0 - 3 * tt0.norm() ** 2) < 1e-8
    tt2 = tt1 / 3.0
    s_vals1 = tt1.sing_vals()
    s_vals2 = tt2.sing_vals()
    for s1, s2 in zip(s_vals1, s_vals2):
        assert ar.do("linalg.norm", s1 - s2 * 3) < 1e-8
    tt1 /= 3.0
    tt3 = tt0.copy(deep=True)
    tt3 -= tt2
    assert (tt0 + (-tt1)).norm() < 1e-8
    assert (tt0 - tt2).norm() < 1e-8
    assert tt3.norm() < 1e-8
    tt1[-1] = ar.do("zeros_like", tt1[-1])
    assert tt1.norm() < 1e-8


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("d", [3, 4, 10])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_ttmlv_arithmetic(backend, r, d, order):
    tt = TensorTrain.random([d] * order, r, backend=backend)

    ttmlv0 = TensorTrainTangentVector.random(
        tt, tt.orthogonalize(mode="r", inplace=False)
    )
    ttmlv1 = ttmlv0 + ttmlv0
    ttmlv1 += ttmlv0
    assert np.abs(ttmlv1 @ ttmlv0 - 3 * ttmlv0.norm() ** 2) < 1e-8
    ttmlv2 = ttmlv1 / 3.0
    for c1, c2 in zip(ttmlv0, ttmlv2):
        assert ar.do("linalg.norm", c1 - c2) < 1e-8
    ttmlv1 /= 3.0
    ttmlv3 = ttmlv0.copy(deep=True)
    ttmlv3 -= ttmlv2
    assert (ttmlv0 + (-ttmlv1)).norm() < 1e-8
    assert (ttmlv0 - ttmlv2).norm() < 1e-8
    assert ttmlv3.norm() < 1e-8
    for i in range(len(ttmlv1)):
        ttmlv1[i] = ar.do("zeros_like", ttmlv1[i])
    ttmlv1.convert_backend("numpy")
    assert ttmlv1.norm() < 1e-8


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("d", [3, 4, 10])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("N", [10, 1000])
@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def testrgrad_sparse(backend, r, d, order, N):
    # Check that rgrad_sparse defines a projection
    tt = TensorTrain.random([d] * order, r, backend=backend)

    idx = random_idx(tt, N, backend=backend)
    y = random_normal((N,), backend=backend)
    idx, y = merge_sum(idx, y)

    ttmlv = tt.rgrad_sparse(y, idx)
    dense_grad = ttmlv.to_tt().dense()
    dense_grad = ar.to_numpy(dense_grad)
    idx_all = np.indices(dense_grad.shape).reshape(len(dense_grad.shape), -1).T
    idx_all = convert_backend(idx_all, backend)
    dense_grad = convert_backend(dense_grad.reshape(-1), backend)

    ttmlv1 = tt.rgrad_sparse(dense_grad, idx_all)
    assert (ttmlv - ttmlv1).norm() < 1e-8


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("d", [3, 4, 10])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def testtml_proj(backend, r, d, order):
    # tt_proj should be a projection
    tt = TensorTrain.random([d] * order, r, backend=backend)
    tt1 = TensorTrain.random([d] * order, r, backend=backend)
    tt2 = tt.tt_proj(tt1)
    tt3 = tt.grad_proj(tt2)
    assert (tt3 - tt2).norm() < 1e-8


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("d", [3, 4, 10])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def testapply_grad(backend, r, d, order):
    # Do a small step twice, confirm this is close step of double step size
    # Also confirm norm actually changes
    tt = TensorTrain.random([d] * order, r, backend=backend)
    tt_r = tt.orthogonalize(mode="r", inplace=False)
    ttmlv = TensorTrainTangentVector.random(tt, tt_r)
    alpha = 1e-6
    tt1 = tt.apply_grad(ttmlv, alpha=alpha)
    ttmlv2 = tt1.grad_proj(ttmlv)
    tt1.apply_grad(ttmlv2, alpha=alpha, inplace=True)
    tt2 = tt.apply_grad(ttmlv, alpha=2 * alpha)

    assert (tt2 - tt1).norm() < 1e-12
    assert (tt2 - tt).norm() > 1e-8


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_increase_rank(backend):
    order = 4
    d = 7
    r = 2
    tt = TensorTrain.random([d] * order, r, backend=backend)
    dense_before = tt.dense()

    # increase all ranks by 2
    tt.increase_rank(2)
    new_rank = tuple(c.shape[0] for c in tt.cores[1:])
    assert new_rank == (4, 4, 4) == tt.tt_rank
    assert ar.do("linalg.norm", tt.dense() - dense_before) < 1e-8
    tt.orthogonalize("l")
    tt.orthogonalize("r")
    new_rank = tuple(c.shape[0] for c in tt.cores[1:])
    assert ar.do("linalg.norm", tt.dense() - dense_before) < 1e-8
    assert new_rank == (4, 4, 4) == tt.tt_rank

    # increase middle rank by 3
    tt.increase_rank(3, 1)
    new_rank = tuple(c.shape[0] for c in tt.cores[1:])
    assert new_rank == (4, 7, 4) == tt.tt_rank
    assert ar.do("linalg.norm", tt.dense() - dense_before) < 1e-8
    tt.orthogonalize("l")
    tt.orthogonalize("r")
    new_rank = tuple(c.shape[0] for c in tt.cores[1:])
    assert ar.do("linalg.norm", tt.dense() - dense_before) < 1e-8
    assert new_rank == (4, 7, 4) == tt.tt_rank
