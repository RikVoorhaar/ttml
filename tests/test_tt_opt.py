import autoray as ar
import numpy as np
import pytest
import sys


from ttml.tensor_train import (
    TensorTrain,
)

from ttml.tt_rlinesearch import (
    TTLS,
    nonmonotone_armijo,
)

from ttml.tt_radam import TensorTrainSGD

from ttml.utils import (
    SUPPORTED_BACKENDS,
    random_idx,
    predict_logit,
)


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("N", (100, 1000))
@pytest.mark.parametrize("cg_method", ("sd", "fr"))
@pytest.mark.parametrize("line_search_method", ("armijo", "wolfe"))
@pytest.mark.parametrize("memory", (1, 3))
@pytest.mark.parametrize(
    "initial_stepsize_method", ("bb1", "bb2", "qopt", "scalar")
)
def test_ttls(
    backend, N, cg_method, line_search_method, memory, initial_stepsize_method
):
    """Check if a small pertubation of TT can be restored"""
    r = 2
    d = 4
    order = 3

    tt0 = TensorTrain.random([d] * order, r, backend=backend)
    tt = tt0 + 1e-6 * TensorTrain.random([d] * order, r, backend=backend)
    tt.round(max_rank=r)

    idx = random_idx(tt0, N, backend=backend)
    y = tt0.gather(idx)

    ttls = TTLS(
        tt,
        y,
        idx,
        cg_method=cg_method,
        line_search_method=line_search_method,
        memory=memory,
        initial_stepsize_method=initial_stepsize_method,
    )
    error_before = (tt0 - ttls.tt).norm()
    for _ in range(3):
        ttls.step()
    error_after = (tt0 - ttls.tt).norm()
    assert error_after < error_before


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("N", (10, 100))
@pytest.mark.parametrize("cg_method", ("sd", "fr"))
@pytest.mark.parametrize("use_sample_weight", (True, False))
@pytest.mark.parametrize("task", ("regression", "classification"))
def test_linesearch_derivatives(
    backend, N, cg_method, use_sample_weight, task
):
    """Compute line search objective and derivatives for a bunch of values. Compare the resulting
    derivatives to estimated derivatives from finite differences, so long as step size is small
    these should be very close."""
    r = 2
    d = 3
    order = 4
    tt0 = TensorTrain.random([d] * order, r, backend=backend)
    # small perturbation
    tt = tt0 + 1e-6 * TensorTrain.random([d] * order, r, backend=backend)
    tt.round(max_rank=r)

    idx = random_idx(tt0, N, backend=backend)
    y = tt0.gather(idx)
    if task == "classification":
        y = predict_logit(y, random=False)

    if use_sample_weight:
        sample_weight = ar.do(
            "random.uniform",
            dtype="float64",
            low=1.0,
            high=2.0,
            size=y.shape,
            like=backend,
        )
    else:
        sample_weight = None
    ttls = TTLS(
        tt, y, idx, task=task, sample_weight=sample_weight, cg_method=cg_method
    )
    alpha_max = 1e-6
    plot_X, phis, der_phis = ttls.plot_linesearch(alpha_max=alpha_max)
    rder_phis = []
    for alpha in plot_X:
        _, derphi = ttls._phi_derphi(alpha, riemannian=True)
        rder_phis.append(derphi)
    rder_phis = np.array(rder_phis)

    findiff = np.diff(phis) / np.diff(plot_X)
    der_phis_mean2 = (der_phis[1:] + der_phis[:-1]) / 2
    rder_phis_mean2 = (rder_phis[1:] + rder_phis[:-1]) / 2

    fin_diff_norm = np.linalg.norm(findiff)
    if fin_diff_norm < 1e-6:
        fin_diff_norm = 1e-6

    assert np.linalg.norm(findiff - der_phis_mean2) / fin_diff_norm < 1e-2
    assert np.linalg.norm(findiff - rder_phis_mean2) / fin_diff_norm < 1e-2


def make_phi(a):
    def phi(t):
        return (a - 2 * t * a) ** 2

    return phi


def make_derphi(a):
    def derphi(t):
        return -4 * a ** 2 * (1 - 2 * t)

    return derphi


@pytest.mark.parametrize("use_alpha0", [True, False])
@pytest.mark.parametrize("memory", [2, 3, 5])
def test_nonmonotone_armijo(use_alpha0, memory):
    a = 1
    phi_history = []
    step_sizes = [1]
    for _ in range(100):
        if use_alpha0:
            alpha0 = 2 * step_sizes[-1]
        else:
            alpha0 = 1
        step_size = nonmonotone_armijo(
            make_phi(a),
            make_derphi(a),
            phi_history=phi_history,
            alpha0=alpha0,
            memory=memory,
        )
        a -= 2 * a * step_size
        phi_history.append(a ** 2)
        step_sizes.append(step_size)
    phi_history = np.array(phi_history)
    max_stack = np.stack(
        [phi_history[i : -(memory - i)] for i in range(memory)]
    )
    max_mem = np.max(max_stack, axis=0)
    assert np.all(phi_history[memory:] < max_mem)


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("batch_size", (10, 100))
@pytest.mark.parametrize("task", ("regression", "classification"))
def test_tt_radam(backend, batch_size, task):
    r = 2
    d = 4
    order = 3
    N = 10000

    tt0 = TensorTrain.random([d] * order, r, backend=backend)
    tt = tt0 + 1e-4 * TensorTrain.random([d] * order, r, backend=backend)
    tt.round(max_rank=r)

    idx = random_idx(tt0, N, backend=backend)
    y = tt0.gather(idx)
    if task == "classification":
        y = predict_logit(y, random=False)

    tt_radam = TensorTrainSGD(
        tt, y, idx, batch_size=batch_size, lr=1e-9, task=task
    )
    error_before = (tt0 - tt_radam.tt).norm()
    for _ in range(5):
        tt_radam.step()
    error_after = (tt0 - tt_radam.tt).norm()
    assert error_after < 2*error_before