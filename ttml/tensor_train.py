from copy import copy, deepcopy

import autoray as ar
import opt_einsum

try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass
from autoray import numpy as np
from numpy import sqrt

from .utils import (
    convert_backend_cores,
    random_normal,
    random_isometry,
    trim_ranks,
)

# TODO: automatic right_core caching


class TensorTrain:
    """
    Implements a tensor train with orthogonalized cores.

    Parameters
    ----------
    cores : list<order 3 tensors>
        List of the TT cores. First core should have shape (1,d0,r0), last has
        shape (r(n-1),dn,1). Last dimension should match first dimension of
        consecutive tensors in the list. The tensors can be `numpy.ndarray`,
        `torch.Tensor` or `tensorflow.EagerTensor`, so long as all the tensors
        are the same type.
    mode : str or int (default: `'l'`)
        The orthogonalization mode. Can be an int, or the string `'l'` or `'r'`,
        which respectively get converted to the `self.order-1` and `0`
    is_orth : bool (default: `False`)
        Whether or not the cores are already orthogonalized. If `False`, the
        cores are orthogonalized during init.

    Attributes
    ----------
    cores : list<order 3 tensors>
        List of all the tensor train cores.
    order : int
        The order of the tensor train (number of cores)
    dims : tuple<int> of length self.order
        The outer dimensions of the tensor train, i.e. the shape of the dense
        tensor represented by the tensor train, or `shape[1]` for each core.
    tt_rank : tuple<int> of length self.order-1
        The tt-rank. This `shape[0]` or `shape[2]` for each core. This tuple
        starts with `cores[0].shape[2]`, and hence does always start/end with 1.
    backend : str
        String encoding the backend of the tensors. This is inferred from
        cores[0].
    """

    def __init__(self, cores, mode="l", is_orth=False):
        self.cores = cores
        self.order = len(cores)
        self.dims = tuple(c.shape[1] for c in cores)
        self.tt_rank = tuple(c.shape[0] for c in cores[1:])
        self.backend = ar.infer_backend(cores[0])

        if mode == "l":
            self.mode = self.order - 1
        elif mode == "r":
            self.mode = 0
        else:
            self.mode = mode
        if mode is not None:
            if not is_orth:
                # Orthogonalize. If inplace it modifies the argument cores
                self.cores = self._orth_cores(mode, inplace=False)
            self.is_orth = True
        else:
            self.is_orth = False

    def orthogonalize(self, mode="l", inplace=True, force_rank=True):
        """Orthogonalize the cores with respect to `mode`.

        Parameters
        ----------
        mode : int or str (default: "l")
            Orthogonalization mode. If "l", defaults to right-most core, if "r"
            to left-most core.
        inplace : bool (default: True)
            If `True` then cores are changed in place and return `None`.
            Otherwise return a TensorTrain object with orthogonalized cores.
        force_rank : bool (default: True)
            If True, check after each step that the rank of the TT hasn't
            lowered. If it has, artificially increase it back by mutliplying by
            random isometry.
        """
        new_cores = self._orth_cores(
            mode=mode, inplace=inplace, force_rank=force_rank
        )
        if not inplace:
            return TensorTrain(new_cores, mode=mode, is_orth=True)

    def _orth_cores(self, mode="l", inplace=True, force_rank=True):
        """Orthogonalize with respect to mode `mode` and return list of new TT
        cores.

        See `orthogonalize` for parameters.
        """

        if mode == "l":
            mu = self.order - 1
        elif mode == "r":
            mu = 0
        else:
            if not isinstance(mode, int):
                raise ValueError(
                    "Orthogonalization mode should be 'l','r' or int"
                )
            mu = mode

        new_cores = [None] * self.order
        if inplace:
            new_cores[mu] = self.cores[mu]
        else:
            new_cores[mu] = deepcopy(self.cores[mu])

        # Orthogonalize to left of mu
        if mu > 0:
            for i in range(mu):
                if inplace:
                    C = self.cores[i]
                else:
                    C = deepcopy(self.cores[i])
                shape = C.shape
                if i > 0:
                    C = np.reshape(C, (shape[0], shape[1] * shape[2]))
                    C = R @ C
                    K = R.shape[0]
                else:
                    K = 1
                C = np.reshape(C, (K * shape[1], shape[2]))
                Q, R = np.linalg.qr(C)
                if (
                    force_rank and R.shape[0] < R.shape[1]
                ):  # detect rank decrease
                    isometry = random_isometry(
                        (R.shape[1], R.shape[0]), self.backend
                    )
                    R = isometry @ R
                    Q = Q @ ar.transpose(isometry)

                new_cores[i] = np.reshape(Q, (K, shape[1], Q.shape[1]))
            if mu == self.order - 1:
                C = new_cores[mu]
                C = np.reshape(C, C.shape[:-1])
                C = R @ C
                Q, R = np.linalg.qr(C)
                new_cores[mu - 1] = ar.do(
                    "einsum", "ijk,kl->ijl", new_cores[mu - 1], Q
                )
                new_cores[mu] = np.reshape(R, R.shape + (1,))
            if mu < self.order - 1:
                C = new_cores[mu]
                shape = C.shape
                C = np.reshape(C, (shape[0], shape[1] * shape[2]))
                C = R @ C
                C = np.reshape(C, (R.shape[0], shape[1], shape[2]))
                new_cores[mu] = C

        # Orthogonalize to the right of mu
        if mu < self.order - 1:
            for i in range(self.order - 1, mu, -1):
                if inplace:
                    C = self.cores[i]
                else:
                    C = deepcopy(self.cores[i])
                shape = C.shape
                if i < self.order - 1:
                    C = np.reshape(C, (shape[0] * shape[1], shape[2]))
                    C = C @ np.transpose(R)
                    K = R.shape[0]
                else:
                    K = 1
                C = np.reshape(C, (shape[0], shape[1] * K))
                Q, R = np.linalg.qr(np.transpose(C))
                if (
                    force_rank and R.shape[0] < R.shape[1]
                ):  # detect rank decrease
                    isometry = random_isometry(
                        (R.shape[1], R.shape[0]), self.backend
                    )
                    R = isometry @ R
                    Q = Q @ ar.transpose(isometry)
                if i == 0:
                    new_cores[i + 1] = ar.do(
                        "einsum", "ji,jkl->ikl", Q, new_cores[i + 1]
                    )
                    new_cores[i] = np.reshape(np.transpose(R), (1,) + R.shape)
                else:
                    new_cores[i] = np.reshape(
                        np.transpose(Q), (Q.shape[1], shape[1], K)
                    )
            if mu == 0:
                C = new_cores[mu]
                C = np.reshape(C, C.shape[1:])
                C = R @ np.transpose(C)
                Q, R = np.linalg.qr(C)
                new_cores[mu + 1] = ar.do(
                    "einsum", "ji,jkl->ikl", Q, new_cores[mu + 1]
                )
                new_cores[mu] = np.reshape(
                    np.transpose(R), (1,) + R.shape[::-1]
                )
            if mu > 0:
                C = new_cores[mu]
                shape = C.shape
                C = np.reshape(C, (shape[0] * shape[1], shape[2]))
                C = C @ np.transpose(R)
                C = np.reshape(C, (shape[0], shape[1], R.shape[0]))
                new_cores[mu] = C

        if inplace:
            self.cores = new_cores
            self.is_orth = True
            self.mode = mu
            self.tt_rank = tuple(c.shape[0] for c in new_cores[1:])

        return new_cores

    def convert_backend(self, backend):
        """Change the backend to target backend inplace."""
        if self.backend == backend:
            pass
        self.cores = convert_backend_cores(self.cores, backend)
        self.backend = backend

    @classmethod
    def from_dense(cls, X):
        raise NotImplementedError

    def dense(self):
        """Contract to dense tensor in left-to-right sweep.

        Returns
        -------
        np.ndarray
            Dense tensor with shape `self.dims`
        """
        C = self.cores[0]
        shape = C.shape
        contracted = np.reshape(C, (shape[0] * shape[1], shape[2]))
        for C in self.cores[1:]:
            shape1 = contracted.shape
            shape2 = C.shape
            contracted = contracted @ np.reshape(
                C, (shape2[0], shape2[1] * shape2[2])
            )
            contracted = np.reshape(
                contracted, (shape1[0] * shape2[1], shape2[2])
            )
        contracted = np.reshape(contracted, self.dims)
        return contracted

    @classmethod
    def random(cls, dims, tt_rank, mode="l", backend="numpy", auto_rank=True):
        """Create random TensorTrain of specified shape and rank.

        Always uses double precision for the tensors.

        Parameters
        ----------
        dims : iterable of ints
        tt_rank : int or iterable of ints
            if int, all tt-ranks will be the same
        mode : "l", "r" or int
        backend : str (default: "numpy")
        autorank : bool (default: `True`)
            if True, automatically losslessly reduce the rank. This option
            should mainly be set to `False` for debugging and testing purposes.
        """

        if isinstance(tt_rank, int):
            tt_rank = [tt_rank] * (len(dims) - 1)
        if auto_rank:
            tt_rank = trim_ranks(dims, tt_rank)
        ranks = [1] + list(tt_rank) + [1]
        cores = []
        for i in range(len(dims)):
            C = random_normal(
                (ranks[i] * dims[i], ranks[i + 1]), backend=backend
            )
            if i > 0:
                C, _ = ar.do("linalg.qr", C)  # QR to create O(1) sing vals
            C = ar.do("reshape", C, (ranks[i], dims[i], ranks[i + 1]))
            cores.append(C)
        tt = cls(cores, mode=mode, is_orth=True)
        if auto_rank:
            tt.orthogonalize(force_rank=False)
        else:
            tt.orthogonalize(force_rank=True)
        return tt

    def gather(self, idx):
        """Gather entries of dense tensor according to indices.

        For each row of `idx` this returns one number. This number is obtained
        by multiplying the slices of each core corresponding to each index (in
        a left-to-right fashion).

        Parameters
        ----------
        idx : np.ndarray<int>
            Array of indices of shape `(len(self.dims), -1)`.

        Returns
        -------
        np.ndarray
            Result of contraction.
        """
        # TODO: use caching for further optimization
        # TODO: improve memory efficiency for large idx
        if self.backend == "numpy":
            return self.fast_gather(idx)
        else:
            gather = ar.do("take", self[0], idx[:, 0], axis=1)
            result = ar.do(
                "reshape", gather[0], (gather.shape[1], gather.shape[2])
            )
            for i in range(1, self.order):
                gather = ar.do("take", self[i], idx[:, i], axis=1)
                result = opt_einsum.contract("ij,jik->ik", result, gather)
            return ar.do("reshape", result, (-1,))

    def fast_gather(self, idx):
        """Faster version of standard gather. Only works with numpy backend"""
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

    def idx_env(self, alpha, idx, num_cores=1, flatten=True):
        """Gather the left and right environment of a TT-core.

        Parameters
        ----------
        - cores : list<tensor of order 3>
        - alpha : int
            left site of (super)core
        - idx : array<int>
            positions of data to gather
        - num_cores : int (default: 1)
            number of cores in the supercore
        - flatten : bool, optional (default: True)
            If True, always flatten result to 2D array. Otherwise result can be
            2D or 3D depending on alpha.
        """

        if (alpha + num_cores > self.order) or (alpha < 0):
            raise ValueError("The value of alpha is out of range")

        N = len(idx)

        # Gather indices of all cores to the left / right of supercore
        left = self[:alpha]
        right = self[alpha + num_cores :]

        left_gather = [
            ar.do("take", left[i], idx[:, i], axis=1) for i in range(len(left))
        ]

        right_gather = [
            ar.do("take", right[i], idx[:, alpha + num_cores + i], axis=1)
            for i in range(len(right))
        ]

        # Contract all the cores to the left / right of the supercore
        if alpha != 0:
            left_env = ar.do(
                "reshape",
                left_gather[0],
                (left_gather[0].shape[1], left_gather[0].shape[2]),
            )
            for M in left_gather[1:]:
                left_env = opt_einsum.contract("ij,jik->ik", left_env, M)

        if alpha != len(self) - num_cores:
            right_env = ar.do(
                "reshape",
                right_gather[-1],
                (right_gather[-1].shape[0], right_gather[-1].shape[1]),
            )
            for M in right_gather[-2::-1]:
                right_env = opt_einsum.contract("jik,ki->ji", M, right_env)

        # Tensor left and right environments of the supercore
        if alpha == 0:
            env = ar.do("transpose", right_env)
            if not flatten:
                env = ar.do("reshape", env, (1,) + env.shape)
        elif alpha == len(self) - num_cores:
            env = left_env
            if not flatten:
                env = ar.do("transpose", env)
                env = ar.do("reshape", env, env.shape + (1,))
        else:
            if flatten:
                env = opt_einsum.contract("bi,jb->bij", left_env, right_env)
                env = ar.do("reshape", env, (N, -1))
            else:
                env = opt_einsum.contract("bi,jb->ibj", left_env, right_env)
        return env

    def __mul__(self, other):
        if not self.is_orth:
            self._orth_cores()
        new_cores = deepcopy(self.cores)
        new_cores[self.mode] *= other
        return TensorTrain(new_cores, mode=self.mode, is_orth=True)

    __rmul__ = __mul__

    def __imul__(self, other):
        if not self.is_orth:
            self._orth_cores()
        self.cores[self.mode] *= other
        return self

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __itruediv__(self, other):
        self.__imul__(1 / other)
        return self

    def __add__(self, other):
        # Take direct sum for each tensor slice
        new_cores = [np.concatenate((self.cores[0], other.cores[0]), axis=2)]
        for C1, C2 in zip(self.cores[1:-1], other.cores[1:-1]):
            r1, d, r2 = C1.shape
            r3, _, r4 = C2.shape
            zeros1 = ar.do(
                "zeros",
                (r1, d, r4),
                like=self.backend,
                dtype=ar.to_backend_dtype("float64", self.backend),
            )
            zeros2 = ar.do(
                "zeros",
                (r3, d, r2),
                like=self.backend,
                dtype=ar.to_backend_dtype("float64", self.backend),
            )
            row1 = np.concatenate((C1, zeros1), axis=2)
            row2 = np.concatenate((zeros2, C2), axis=2)
            new_cores.append(np.concatenate((row1, row2), axis=0))
        new_cores.append(
            np.concatenate((self.cores[-1], other.cores[-1]), axis=0)
        )
        new_tt = TensorTrain(new_cores, is_orth=False)
        new_tt.orthogonalize(force_rank=False)
        return new_tt

    def __iadd__(self, other):
        new_tt = self.__add__(other)
        return new_tt
        # self.cores = new_tt.cores
        # self.mode = new_tt.mode
        # return self

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        return self + (-1) * other

    def __isub__(self, other):
        new_tt = self.__sub__(other)
        return new_tt

    def __len__(self):
        return self.order

    def __getitem__(self, index):
        return self.cores[index]

    def __setitem__(self, index, data):
        self.cores[index] = data

    def __repr__(self):
        return (
            f"<TensorTrain of order {self.order} "
            f"with outer dimensions {self.dims}, TT-rank "
            f"{self.tt_rank}, and orthogonalized at mode {self.mode}>"
        )

    def round(self, max_rank=None, eps=None, inplace=True):
        """
        Truncate the tensor train.

        * If ``max_rank`` is specified, truncate to this rank.
        * If ``eps`` is specified, truncate with precision ``eps`` (relative
          to largest singular value of each unfolding)

        Parameters
        ----------
        max_rank : int or tuple<int> (optional)
        eps : float (optional)
        inplace : bool
            If True, round the tensor train in place. Otherwise performI
            truncation on a copy.

        Returns
        -------
        TensorTrain
        """
        if inplace:
            tt = self
        else:
            tt = self.copy()

        if max_rank is None and eps is None:
            raise ValueError(
                "At least one of `max_rank` or `eps` has to be specified"
            )

        # scale epsilon relative to norm
        if eps is not None:
            eps *= tt.norm()

        if isinstance(max_rank, int):
            max_rank = [max_rank] * tt.order

        # sweep left to right, last core there's nothing to do
        tt.orthogonalize(mode="r")
        for i in range(tt.order - 1):
            # truncated SVD
            C = tt.cores[i]
            shape = C.shape
            U, S, V = np.linalg.svd(
                np.reshape(C, (shape[0] * shape[1], shape[2]))
            )
            if max_rank is not None:
                S = S[: max_rank[i]]
            if eps is not None:
                if S[0] > eps:
                    S = S[S > eps]
                else:
                    S = S[:1]
            r = len(S)
            tt.cores[i] = np.reshape(U[:, :r], (shape[0], shape[1], r))

            # update next core
            SV = np.diag(S) @ V[:r, :]
            next_core = tt.cores[i + 1]
            next_shape = next_core.shape
            next_core = np.reshape(
                next_core, (next_shape[0], next_shape[1] * next_shape[2])
            )
            next_core = SV @ next_core
            tt.cores[i + 1] = np.reshape(
                next_core, (r, next_shape[1], next_shape[2])
            )

        tt.mode = tt.order - 1
        tt.tt_rank = tuple(c.shape[0] for c in tt.cores[1:])

        return tt

    def increase_rank(self, inc, i=None):
        """Increase the rank of the edge between core `i` and `i+1` by `inc`.

        This operation leaves the dense tensor invariant.
        To instead decrease the rank, use `truncate`."""

        # Increase the rank of all nodes
        if i is None:
            for i in range(len(self.tt_rank)):
                self.increase_rank(inc, i)
            return self

        # Random isometry of shape (r+inc, r)
        r = self.tt_rank[i]
        A = ar.do(
            "random.normal",
            size=(r + inc, r),
            like=self.backend,
            dtype="float64",
        )
        Q, _ = ar.do("linalg.qr", A)

        # Apply isometry
        self[i] = ar.do("einsum", "ijk,lk->ijl", self[i], Q)
        self[i + 1] = ar.do("einsum", "ij,jkl->ikl", Q, self[i + 1])

        # Update rank information
        new_tt_rank = list(self.tt_rank)
        new_tt_rank[i] += inc
        self.tt_rank = tuple(new_tt_rank)

        return self

    def sing_vals(self):
        """Compute singular values of each unfolding.

        As a side effect the TT is left-orthogonalized. One array of size
        ``self.tt_rank[i]`` is returned for each core ``i``.
        """

        if self.mode != 0:
            self.orthogonalize(mode=0)

        sing_vals = []

        for i in range(self.order - 1):
            #  SVD
            C = self.cores[i]
            shape = C.shape
            U, S, V = np.linalg.svd(
                np.reshape(C, (shape[0] * shape[1], shape[2]))
            )
            sing_vals.append(ar.to_numpy(S))
            self.cores[i] = np.reshape(U, shape)

            # update next core
            SV = np.diag(S) @ V
            next_core = self.cores[i + 1]
            next_shape = next_core.shape
            next_core = np.reshape(
                next_core, (next_shape[0], next_shape[1] * next_shape[2])
            )
            next_core = SV @ next_core
            self.cores[i + 1] = np.reshape(next_core, next_shape)

        self.mode = self.order - 1
        self.tt_rank = tuple(c.shape[0] for c in self.cores[1:])

        return sing_vals

    def rgrad_sparse(self, grad, idx):
        """Project sparse euclidean gradient to tangent space.

        Parameters
        ----------
        grad : array<float64>
            Array containing the values of the sparse gradient.
        idx : array<int64> of shape `(len(grad),self.order)`
            Array containing the indices of the dense tensor corresponding to
            the values of the sparse euclidean gradient

        Returns
        -------
        TensorTrainTangentVector
        """
        if self.mode != self.order - 1:
            self.orthogonalize()
        right_cores = self._orth_cores(mode="r", inplace=False)
        left_cores = self.cores
        for C1, C2 in zip(left_cores, right_cores):
            assert C1.shape == C2.shape, (
                [C.shape for C in left_cores],
                [C.shape for C in right_cores],
            )

        N = len(idx)
        backend = self.backend

        # Compute left vectors
        left_vectors = [None] * self.order
        for mu in range(self.order - 1):
            left_vectors[mu] = ar.do(
                "zeros",
                (N, self.tt_rank[mu]),
                like=backend,
                dtype=ar.to_backend_dtype("float64", backend),
            )
            for i in range(self.dims[mu]):
                inds = ar.do("where", idx[:, mu] == i)[0]
                if len(inds) == 0:
                    continue
                if backend == "tensorflow":
                    inds = ar.do("reshape", inds, (-1, 1))
                if mu == 0:
                    update = ar.do(
                        "stack", [left_cores[0][0, i, :]] * len(inds)
                    )
                    if backend == "tensorflow":
                        left_vectors[mu] = tf.tensor_scatter_nd_update(
                            left_vectors[mu], inds, update
                        )
                    else:
                        left_vectors[mu][inds] = update
                else:
                    update = (
                        ar.do("take", left_vectors[mu - 1], inds, axis=0)
                        @ left_cores[mu][:, i, :]
                    )
                    if backend == "tensorflow":
                        update = ar.do(
                            "reshape",
                            update,
                            (len(inds), left_vectors[mu].shape[1]),
                        )
                        left_vectors[mu] = tf.tensor_scatter_nd_update(
                            left_vectors[mu], inds, update
                        )
                    else:
                        left_vectors[mu][inds] = update

        # Compute right vectors
        right_vectors = [None] * self.order
        for mu in range(self.order - 1, 0, -1):
            right_vectors[mu] = ar.do(
                "zeros",
                (N, self.tt_rank[mu - 1]),
                like=backend,
                dtype=ar.to_backend_dtype("float64", backend),
            )
            for i in range(self.dims[mu]):
                inds = ar.do("where", idx[:, mu] == i)[0]
                if len(inds) == 0:
                    continue
                if backend == "tensorflow":
                    inds = ar.do("reshape", inds, (-1, 1))
                if mu == self.order - 1:
                    update = ar.do(
                        "stack", [right_cores[mu][:, i, 0]] * len(inds)
                    )
                    if backend == "tensorflow":
                        right_vectors[mu] = tf.tensor_scatter_nd_update(
                            right_vectors[mu], inds, update
                        )
                    else:
                        right_vectors[mu][inds] = update
                else:
                    update = ar.do(
                        "take", right_vectors[mu + 1], inds, axis=0
                    ) @ ar.transpose(right_cores[mu][:, i, :])
                    if backend == "tensorflow":
                        update = ar.do(
                            "reshape",
                            update,
                            (len(inds), right_vectors[mu].shape[1]),
                        )
                        right_vectors[mu] = tf.tensor_scatter_nd_update(
                            right_vectors[mu], inds, update
                        )
                    else:
                        right_vectors[mu][inds] = update

        # Compute the gradient cores
        grad_cores = [[None] * d for d in self.dims]
        for mu in range(self.order - 1, -1, -1):
            for i in range(self.dims[mu]):
                inds = ar.do("where", idx[:, mu] == i)[0]
                if len(inds) == 0:
                    if mu == 0:
                        grad_cores[mu][i] = ar.do(
                            "zeros",
                            (1, self.tt_rank[0]),
                            like=backend,
                            dtype=ar.to_backend_dtype("float64", backend),
                        )
                    elif mu == self.order - 1:
                        grad_cores[mu][i] = ar.do(
                            "zeros",
                            (self.tt_rank[-1], 1),
                            like=backend,
                            dtype=ar.to_backend_dtype("float64", backend),
                        )
                    else:
                        grad_cores[mu][i] = ar.do(
                            "zeros",
                            (self.tt_rank[mu - 1], self.tt_rank[mu]),
                            like=backend,
                            dtype=ar.to_backend_dtype("float64", backend),
                        )
                    continue
                Z = ar.do("take", grad, inds, axis=0)
                if mu > 0:
                    U = ar.do("take", left_vectors[mu - 1], inds, axis=0)
                if mu < self.order - 1:
                    V = ar.transpose(
                        ar.do("take", right_vectors[mu + 1], inds, axis=0)
                    )
                if mu == self.order - 1:
                    Z = ar.reshape(Z, (1, -1))
                    G = ar.do("reshape", Z @ U, (-1, 1))
                elif mu == 0:
                    Z = ar.reshape(Z, (-1, 1))
                    G = ar.do("reshape", V @ Z, (1, -1))
                else:
                    G = ar.do("einsum", "ji,j,kj->ik", U, Z, V)
                grad_cores[mu][i] = G
        grad_cores = [ar.do("stack", G_list, axis=1) for G_list in grad_cores]

        # Apply gauge conditions to the gradient cores
        self.apply_gradient_gauge_conditions(left_cores, grad_cores)

        return TensorTrainTangentVector(grad_cores, left_cores, right_cores)

    def apply_gradient_gauge_conditions(self, left_cores, grad_cores):
        """Apply gauge conditions to gradient cores inplace"""
        for mu in range(self.order - 1):
            r1, r2, r3 = left_cores[mu].shape
            U = np.reshape(left_cores[mu], (r1 * r2, r3))
            grad_cores[mu] -= np.reshape(
                U
                @ (np.transpose(U) @ np.reshape(grad_cores[mu], (r1 * r2, r3))),
                (r1, r2, r3),
            )

    def grad_proj(self, tangent_vector, right_cores=None):
        """Project TensorTrainTangentVector to tangent space.

        This implements parallel transport of `tangent_vector` to this point.
        """
        return self.tt_proj(
            tangent_vector.to_tt(round=False), right_cores=right_cores
        )

    def tt_proj(self, tt, right_cores=None, proj_U=True):
        """Project a TT to tangent space of self.

        This needs both left- and right-orthogonal cores. If `right_cores` is
        not specified, the right cores are computed first."""

        if self.mode != self.order - 1:
            self.orthogonalize()
        left_cores = self.cores
        if right_cores is None:
            right_cores = self._orth_cores(mode="r", inplace=False)

        # List of partial contractions with left/right-orthogonal cores and tt
        right = contract_cores(right_cores, tt, "RL", store_parts=True)
        left = contract_cores(left_cores, tt, "LR", store_parts=True)

        # Project grad cores using left and right environments
        grad_cores = []
        grad_cores.append(opt_einsum.contract("ijk,bk->ijb", tt[0], right[1]))
        for i in range(1, self.order - 1):
            grad_cores.append(
                opt_einsum.contract(
                    "ai,ijk,bk->ajb",
                    left[i - 1],
                    tt[i],
                    right[i + 1],
                )
            )
        grad_cores.append(opt_einsum.contract("ai,ijk->ajk", left[-2], tt[-1]))

        # Apply gauge conditions
        self.apply_gradient_gauge_conditions(left_cores, grad_cores)

        return TensorTrainTangentVector(grad_cores, left_cores, right_cores)

    def apply_grad(self, tangent_vector, alpha=1.0, round=True, inplace=False):
        """Compute retract of tangent vector.

        Parameters
        ----------
        tangent_vector : TensorTrainTangentVector
            Tangent vector, assumed to lie in tangent space at current point.
        alpha : float (default: 1.0)
            Stepsize of retract. Using this parameter is equivalent but more
            efficient to supplying `alpha*tangent_vector` as first argument.
        round : bool (default: True)
            After applying tangent vector we end up with a TT of double the
            tt-rank. This bool controls whether to project back to original
            tt-rank, which is necessary if we want to compute the retract.
        inplace : bool (default: False)
            If false, return a new TensorTrain, otherwise update this TT inplace

        TODO: implement efficient reorthogonalization.
        """
        left_cores = tangent_vector.left_cores
        right_cores = tangent_vector.right_cores
        grad_cores = tangent_vector.grad_cores

        # Formula of Steinlechner "Riemannian Optimization for High-Dimensional
        # Tensor Completion" (published version), end of page 10.
        new_cores = [
            np.concatenate((alpha * grad_cores[0], left_cores[0]), axis=2)
        ]
        for U, V, dU in zip(
            left_cores[1:-1], right_cores[1:-1], grad_cores[1:-1]
        ):
            first_row = np.concatenate((V, np.zeros_like(V)), axis=2)
            second_row = np.concatenate((alpha * dU, U), axis=2)
            new_cores.append(np.concatenate((first_row, second_row), axis=0))
        new_cores.append(
            np.concatenate(
                (right_cores[-1], left_cores[-1] + alpha * grad_cores[-1]),
                axis=0,
            )
        )

        new_tt = TensorTrain(new_cores)
        if round:
            new_tt.round(max_rank=self.tt_rank)

        if inplace:
            self.cores = new_tt.cores
            self.tt_rank = new_tt.tt_rank
            self.mode = new_tt.mode
            self.is_orth = new_tt.is_orth
        else:
            return new_tt

    def norm(self):
        if not self.is_orth:
            self.orthogonalize()
        return np.linalg.norm(self.cores[self.mode])

    def dot(self, other):
        """Compute dot product with other TT with same outer dimensions"""
        return contract_cores(self.cores, other.cores)

    def __matmul__(self, other):
        return self.dot(other)

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def num_params(self):
        return sum([np.prod(C.shape) for C in self.cores])


class TensorTrainTangentVector:
    """Class for storing a tangent vector to TT manifold.

    A tangent vector at a point on the TT-manifold is a list of cores with the
    same shape as the TT, satisfying a certain gauge condition. This class
    stores both the left-orthogonal and right orthogonal cores of the original
    TT. This is because both are needed for most computations.

    parameters
    ----------
    grad_cores : list<order 3 tensors>
        list of first-order variation cores
    left_cores : list<order 3 tensors>
        list of left-orthogonal cores
    right_cores : list<order 3 tensors>
        list of right-orthogonal cores"""

    def __init__(self, grad_cores, left_cores, right_cores):

        self.grad_cores = grad_cores
        self.left_cores = left_cores
        self.right_cores = right_cores
        self.backend = ar.infer_backend(grad_cores[0])
        self.order = len(grad_cores)
        self.tt_rank = tuple(c.shape[0] for c in grad_cores[1:])
        self.dims = tuple(c.shape[1] for c in grad_cores)

    def inner(self, other):
        """Compute inner product between two tangent vectors.

        Due to orthogonalization this is just inner product of first-order
        variations"""
        result = 0.0
        for core1, core2 in zip(self.grad_cores, other.grad_cores):
            if self.backend != "tensorflow":
                result += ar.do(
                    "dot", np.reshape(core1, (-1,)), np.reshape(core2, (-1,))
                )
            else:
                result += ar.do(
                    "tensordot",
                    np.reshape(core1, (-1,)),
                    np.reshape(core2, (-1,)),
                    1,
                )
        return result

    def __matmul__(self, other):
        return self.inner(other)

    def norm(self):
        result = 0
        for c in self.grad_cores:
            result += ar.do("linalg.norm", c) ** 2
        return ar.do("sqrt", result)

    @classmethod
    def random(cls, left_cores, right_cores):
        """Random tangent vector with unit norm gradients"""
        order = len(left_cores)
        backend = ar.infer_backend(left_cores[0])
        grad_cores = []
        for i in range(order):
            # C = ar.do("random.normal", size=left_cores[i].shape, like=backend)
            C = random_normal(left_cores[i].shape, backend=backend)
            C = C / (sqrt(order) * np.linalg.norm(C))
            grad_cores.append(C)

        # Poject to range of U
        for i in range(order - 1):
            Y_mat = grad_cores[i]
            shape = Y_mat.shape
            Y_mat = np.reshape(grad_cores[i], (shape[0] * shape[1], shape[2]))
            U_mat = left_cores[i]
            U_mat = np.reshape(U_mat, (shape[0] * shape[1], shape[2]))
            Y_mat -= U_mat @ (np.transpose(U_mat) @ Y_mat)
            grad_cores[i] = np.reshape(Y_mat, shape)
        return cls(grad_cores, left_cores, right_cores)

    def to_tt(self, round=False):
        """Convert to TensorTrain.

        If `round=True` then round to TT of rank same TT-rank as point, if
        `round=False` it will have double the TT-rank of the point."""

        new_cores = [
            np.concatenate((self.grad_cores[0], self.left_cores[0]), axis=2)
        ]
        for U, V, dU in zip(
            self.left_cores[1:-1],
            self.right_cores[1:-1],
            self.grad_cores[1:-1],
        ):
            first_row = np.concatenate((V, np.zeros_like(U)), axis=2)
            second_row = np.concatenate((dU, U), axis=2)
            new_cores.append(np.concatenate((first_row, second_row), axis=0))
        new_cores.append(
            np.concatenate(
                (self.right_cores[-1], self.grad_cores[-1]),
                axis=0,
            )
        )

        new_tt = TensorTrain(new_cores)
        if round:
            new_tt.round(max_rank=self.tt_rank)
        return new_tt

    def to_eucl(self, idx):
        """Return sparse Euclidean gradient with indices `idx`.

        *NB: `idx` should not contain any repeated indices.*"""
        return self.to_tt().gather(idx)

    def convert_backend(self, backend):
        """Change the backend to target backend inplace."""
        if self.backend == backend:
            pass
        self.grad_cores = convert_backend_cores(self.grad_cores, backend)
        self.left_cores = convert_backend_cores(self.left_cores, backend)
        self.right_cores = convert_backend_cores(self.right_cores, backend)
        self.backend = backend

    def __repr__(self):
        return (
            f"<TensorTrainTangentVector of order {self.order}, "
            f"outer dimensions {self.dims}, and TT-rank "
            f"{self.tt_rank}>"
        )

    def __getitem__(self, index):
        return self.grad_cores[index]

    def __setitem__(self, index, data):
        self.grad_cores[index] = data

    def __mul__(self, other):
        new_grad_cores = [C * other for C in self.grad_cores]
        return TensorTrainTangentVector(
            new_grad_cores, self.left_cores, self.right_cores
        )

    __rmul__ = __mul__

    def __imul__(self, other):
        self.grad_cores = [C * other for C in self.grad_cores]
        return self

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __itruediv__(self, other):
        self.__imul__(1 / other)
        return self

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        return self + (-1) * other

    def __isub__(self, other):
        self.__iadd__(-other)
        return self

    def __add__(self, other):
        new_grad_cores = [
            C1 + C2 for C1, C2 in zip(self.grad_cores, other.grad_cores)
        ]
        return TensorTrainTangentVector(
            new_grad_cores, self.left_cores, self.right_cores
        )

    def __iadd__(self, other):
        self.grad_cores = [
            C1 + C2 for C1, C2 in zip(self.grad_cores, other.grad_cores)
        ]
        return self

    def __len__(self):
        return self.order

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)


def contract_cores(
    cores1, cores2, dir="LR", upto=None, store_parts=False, return_float=True
):
    """Compute contraction of one list of TT cores with another.

    Cores should be of outer dimensions If `dir=LR`, then first cores should
    have first shape 1, if `dir=RL` then last shape of last core should be 1.

    Parameters
    ----------
    cores1 : list of TT cores.
    cores2 : list of TT cores.
    dir : str, default='LR'
        Direction of contraction; LR (left-to-right), RL (right-to-left)
    upto : int, optional
        Contract only up to this mode
    store_parts : bool, default=False
        If `True`, return list of all intermediate contractions.
    return_float : bool, default=True
        If `True` and the result has total dimension 1, then compress result
        down to a float. Ignored if `store_parts=True`.
    """
    result_list = []
    if dir == "LR":
        result = opt_einsum.contract("ijk,ajc->iakc", cores1[0], cores2[0])

        result = ar.do(
            "reshape", result, (cores1[0].shape[-1], cores2[0].shape[-1])
        )
        for core1, core2 in zip(cores1[1:upto], cores2[1:upto]):
            if store_parts:
                result_list.append(result)
            result = opt_einsum.contract("ij,ika,jkb->ab", result, core1, core2)

        total_dim = 1
        for d in result.shape:  # if result is number, reformat to float.
            total_dim *= d
        if return_float and d == 1 and not store_parts:
            result = float(np.reshape(result, (-1,)))
        if store_parts:
            result_list.append(result)
            return result_list
        else:
            return result
    elif dir == "RL":
        result = opt_einsum.contract("ijk,ajc->iakc", cores1[-1], cores2[-1])

        result = ar.do(
            "reshape", result, (cores1[-1].shape[0], cores2[-1].shape[0])
        )
        for core1, core2 in zip(cores1[-2:upto:-1], cores2[-2:upto:-1]):
            if store_parts:
                result_list.append(result)
            result = opt_einsum.contract("ab,ika,jkb->ij", result, core1, core2)

        total_dim = 1
        for d in result.shape:  # if result is number, reformat to float.
            total_dim *= d
        if return_float and d == 1 and not store_parts:
            result = float(np.reshape(result, (-1,)))
        if store_parts:
            result_list.append(result)
            return result_list[::-1]  # reverse result list for RL direction
        else:
            return result
    else:
        raise ValueError(f"Unknown direction '{dir}'")
