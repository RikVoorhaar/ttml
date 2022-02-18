import autoray as ar
import numpy as np
import tensorflow as tf


def completion_error(X, Y, M, mask):
    """
    Relative error of the low-rank matrix completion problem mask*(X@Y) - mask*(M)
    """
    return np.linalg.norm(mask * (X @ Y) - mask * (M)) / np.linalg.norm(
        mask * M
    )


def lstsq_reg(A, b, rcond_max=1000):
    """Solve least-squares problem by regularized pseudo-inverse.

    We throw away all singular values that are at least `rcond_max` times smaller than the largest
    singular value."""
    U, S, V = ar.do("linalg.svd", A)
    S = S[S >= S[0] / rcond_max]
    r = len(S)
    x = ar.do("transpose", V)[:, :r] @ (
        ar.do("diag", S ** -1) @ (ar.do("transpose", U)[:r, :] @ b)
    )
    return x


def complete_matrix_orth(
    M, mask, r, direction, X0=None, Y0=None, maxiter=100, tol=1e-2
):
    """
    Compute low rank matrix completion of `M` with observed values `mask` and matrix rank `r`.

    We compute mask*(XY) ~ mask*M. If `direction='l'` then Y is orthogonal, if `direction='r'`
    then X is orthogonal.
    If X0, or Y0 are supplied they will be used during iteration. Note that they are overwritten.

    If the relative change in error is less than `tol`, iteration is stopped. The default `tol=1e-2`
    halts if the error changed by less than 1% after one iteration.
    TODO: backend agnostic
    """
    Z = M * mask
    if X0 is None:
        X = np.random.normal(size=(M.shape[0], r))
    else:
        X = X0
    if Y0 is None:
        Y = np.random.normal(size=(r, M.shape[1]))
    else:
        Y = Y0

    last_error = np.inf

    for _ in range(maxiter):
        if direction == "l":
            U, _, V = np.linalg.svd(X.T @ Z, full_matrices=False)
            Y = U @ V
            X = Z @ Y.T
        elif direction == "r":
            U, _, V = np.linalg.svd(Z @ Y.T, full_matrices=False)
            X = U @ V
            Y = X.T @ Z
        else:
            raise ValueError("Direction should be either 'l' or 'r'")

        Z = X @ Y
        Z = Z + mask * (M - Z)
        new_error = completion_error(X, Y, M, mask)
        if (last_error - new_error) / new_error < tol:
            break
        last_error = new_error
    return X, Y, new_error


def dmrg_step(tt, alpha, idx, y, direction="l", rcond_max=1000):
    """One step of the DMRG algorithm

    Parameters
    ----------
    - cores: List of TT cores
    - alpha: Index of the left TT core (other core is alpha+1)
    - idx: Indices corresponding to the data. Array of shape (len(y),len(cores))
    - y: Training data values
    - direction: Direction of sweep. If "l" then right core is orthogonal, if "r" then left core is
      orthogonal
    """

    # Shape of left / right core
    rank_L, thresh_L, rank_C = tt[alpha].shape
    _, thresh_R, rank_R = tt[alpha + 1].shape

    env = tt.idx_env(alpha, idx, num_cores=2)

    mask = ar.do(
        "zeros",
        (rank_L, thresh_L, thresh_R, rank_R),
        like=tt.backend,
        dtype="float64",
    )
    dmrg_mat = ar.do(
        "zeros",
        (rank_L, thresh_L, thresh_R, rank_R),
        like=tt.backend,
        dtype="float64",
    )

    # TODO: Find a way to parallelize this
    # For each pair of indices, compute the slice of the DMRG matrix
    for i in range(thresh_L):
        for j in range(thresh_R):
            # Find all relevant data indices, do nothing if there are none
            inds = ar.do(
                "where", (idx[:, alpha] == i) & (idx[:, alpha + 1] == j)
            )
            inds = ar.do("reshape", inds, (-1,))
            if len(inds) == 0:
                continue
            # Form and solve linear problem, store result
            A = ar.do("take", env, inds, 0)
            b = ar.do("reshape", ar.do("take", y, inds, 0), (-1, 1))
            x = lstsq_reg(A, b, rcond_max=rcond_max)
            mask[:, i, j, :] = 1
            dmrg_mat[:, i, j, :] = x.reshape((rank_L, rank_R))

    # Solve matrix completion problem to get left and right core
    dmrg_mat = ar.do(
        "reshape", dmrg_mat, (rank_L * thresh_L, thresh_R * rank_R)
    )
    mask = ar.do("reshape", mask, (rank_L * thresh_L, thresh_R * rank_R))
    left_core = ar.do("reshape", tt[alpha], (rank_L * thresh_L, rank_C))
    right_core = ar.do("reshape", tt[alpha + 1], (rank_C, thresh_R * rank_R))
    left_core, right_core, _ = complete_matrix_orth(
        dmrg_mat, mask, rank_C, direction, X0=left_core, Y0=right_core
    )

    return ar.do("reshape", left_core, (rank_L, thresh_L, rank_C)), ar.do(
        "reshape", right_core, (rank_C, thresh_R, rank_R)
    )


def als_step(tt, alpha, idx, y, direction="c", debug=False):
    """One step of the ALS algorithm

    Parameters
    ----------
    - cores: List of TT cores
    - alpha: Index of the core
    - idx: Indices corresponding to the data. Array of shape (len(y),len(cores))
    - y: Training data values
    - direction: Direction of sweep. If 'l' then take left-orthogonalize with RQ decomposition (for
      right-to-left sweep), if 'r' then right-orthogonalize with QR decomposition (for left-to-right
      sweep), and if 'c' output as-is (for last step in either sweep)
    - debug: bool (default: False)
        if True, output debug information; namely the linear problems for each
        slice.
    """

    # Gather indices of all cores to the left / right of the core
    rank_left, n_thresh, rank_right = tt[alpha].shape

    env = tt.idx_env(alpha, idx, num_cores=1)

    # for each threshold, compute associated slice of the matrix and form matrix
    n_thresh = tt[alpha].shape[1]
    all_x = []
    all_problems = []
    for i in range(n_thresh):
        inds = ar.do("reshape", ar.do("where", (idx[:, alpha] == i)), [-1])
        A = ar.do("take", env, inds, 0)
        b = ar.do("reshape", ar.do("take", y, inds, 0), [-1, 1])
        all_problems.append([A, b])
        x = lstsq_reg(A, b)
        all_x.append(x)
    new_core = ar.do("stack", all_x)
    new_core = ar.do("reshape", new_core, [n_thresh, rank_left, rank_right])
    new_core = ar.do("transpose", new_core, [1, 0, 2])

    if direction == "c":  # for end of the sweep
        if debug:
            return new_core, all_problems
        else:
            return new_core
    if direction == "r":  # for left-to-right sweep
        reshaped = ar.do(
            "reshape", new_core, [rank_left * n_thresh, rank_right]
        )
        Q, _ = ar.do("linalg.qr", reshaped)
        Q = ar.do("reshape", Q, [rank_left, n_thresh, rank_right])
        if debug:
            return Q, all_problems
        else:
            return Q
    elif direction == "l":  # for right-to-left sweep
        reshaped = tf.reshape(new_core, [rank_left, n_thresh * rank_right])
        reshaped = tf.transpose(reshaped)
        Q, _ = ar.do("linalg.qr", reshaped)
        Q = ar.do("reshape", tf.transpose(Q), [rank_left, n_thresh, rank_right])
        if debug:
            return Q, all_problems
        else:
            return Q
