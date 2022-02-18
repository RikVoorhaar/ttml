# %%
from ttml.tensor_train import TensorTrain
import numpy as np

n_dims = 5
dims = tuple(range(5,5+n_dims))
tot_dim = np.prod(dims)
tt_rank = tuple(range(3+n_dims,3,-1))
tt = TensorTrain.random(dims, tt_rank)
# tt.gather()
permu = np.random.permutation(tot_dim)
idx = np.stack(np.unravel_index(permu, dims)).T

# assert correctness of gather statement
assert np.linalg.norm(tt.dense().reshape(-1)[permu] - tt.gather(idx)) < 1e-10

%time tt.gather(inds)

# %%
def fast_gather(self, idx):
    idx = idx.T
    N = idx.shape[1]
    result = np.take(self[0].reshape(self[0].shape[1:]), idx[0], axis=0)
    for i in range(1, self.order):
        r = self[i].shape[2]
        next_step = np.zeros((N, r))
        for j in range(self.dims[i]):
            idx_mask = np.where(idx[i] == j)
            mat = self[i][:, j, :]
            next_step[idx_mask] = result[idx_mask] @ mat
        result = next_step
    return result.reshape(-1)
assert np.linalg.norm(tt.dense().reshape(-1)[permu] - fast_gather(tt, idx)) < 1e-10

# %%
%load_ext heat
# %%
%%heat

from ttml.tensor_train import TensorTrain
import numpy as np
n_dims = 5
dims = tuple(range(5,5+n_dims))
tot_dim = np.prod(dims)
tt_rank = tuple(range(3+n_dims,3,-1))
tt = TensorTrain.random(dims, tt_rank)
# tt.gather()
permu = np.random.permutation(tot_dim)
idx = np.stack(np.unravel_index(permu, dims)).T

for _ in range(100):
    fast_gather(tt, idx)
# %%
