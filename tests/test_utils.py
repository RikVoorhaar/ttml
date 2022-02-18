import pytest
import autoray as ar

from ttml.utils import (
    SUPPORTED_BACKENDS,
    convert_backend,
    merge_sum,
    random_idx,
    random_normal,
)

from ttml.tensor_train import TensorTrain


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("N", (10, 1000))
def test_merge_sum(backend, N):
    tt = TensorTrain.random([4] * 5, 3, backend=backend)
    idx1 = random_idx(tt, N, backend=backend)
    y1 = tt.gather(idx1)

    idx2 = ar.do("concatenate", [idx1, idx1], 0)
    y2 = tt.gather(idx2)

    red_idx1, red_y1 = merge_sum(idx1, y1)
    red_idx2, red_y2 = merge_sum(idx2, y2)
    assert ar.do("linalg.norm", red_y1 * 2 - red_y2) < 1e-8
    assert (
        ar.do("linalg.norm", ar.to_numpy(red_idx1) - ar.to_numpy(red_idx2))
        < 1e-8
    )
