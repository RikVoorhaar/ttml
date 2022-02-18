import os
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from .tensor_train import TensorTrain, TensorTrainTangentVector
from .utils import convert_backend_cores


def fix_shape(cores):
    if (len(cores[0].shape)) == 2:
        cores[0] = np.reshape(cores[0], (1,) + cores[0].shape)
    d = len(cores) - 1
    if (len(cores[d].shape)) == 2:
        cores[d] = np.reshape(cores[d], cores[d].shape + (1,))


def ndarray_to_matlab(x, eng):
    "Convert Numpy ndarray (or any object that can be coerced into it) to Matlab object."
    _, f = tempfile.mkstemp(suffix=".mat")
    savemat(f, {"x": x})
    x_mat = eng.load(f, "x")["x"]
    os.remove(f)
    return x_mat


def matlab_to_ndarray(x, eng):
    "Convert Matlab object to a Numpy ndarray."
    _, f = tempfile.mkstemp(suffix=".mat")
    eng.Object_to_Py("../matlab_files/mat.mat", x, nargout=0)
    x_py = loadmat("../matlab_files/mat.mat")["x"]
    os.remove(f)
    return x_py


def load_matlab_tt(fname, is_orth=True, mode="l", backend="numpy"):
    """Load a TT from a MATLAB file.

    The file should contain a variable 'U', which should be a cell of TT-cores. Dimension and rank
    information is inferred.

    This assumes that the TTeMPS is left-orthogonal, if not, specify `is_orth=False`

    BUG: Shape of rank-1 cores is wrong."""

    matlab_tt = loadmat(fname, simplify_cells=True)
    cores = matlab_tt["U"]
    fix_shape(cores)
    tt = TensorTrain(cores, is_orth=is_orth, mode=mode)
    if backend != "numpy":
        tt.convert_backend(backend)
    return tt


def save_matlab_tt(tt, fname):
    """Save TensorTrain to .mat file"""

    cores = tt.cores
    if tt.backend != "numpy":
        cores = convert_backend_cores(cores, "numpy")
    savemat(fname, {"U": np.array(cores, dtype="object")})


def tt_to_matlab(tt, eng):
    """Load TensorTrain into matlab engine as TTeMPS"""

    _, f = tempfile.mkstemp(suffix=".mat")
    save_matlab_tt(tt, f)
    U = eng.Py_to_TTeMPS(f)
    os.remove(f)
    return U


def matlab_to_tt(ttemps, eng, is_orth=True, backend="numpy", mode="l"):
    """Load matlab.object representing TTeMPS into Python as TT"""

    _, f = tempfile.mkstemp(suffix=".mat")
    eng.TTeMPS_to_Py(f, ttemps, nargout=0)
    tt = load_matlab_tt(f, is_orth=is_orth, mode=mode, backend=backend)

    return tt


def load_matlab_ttmlv(fname, backend="numpy"):
    """Load a TT from a MATLAB file.

    The file should contain a variable 'vect_cell', which should be a cell of length 3 containing
    cells of TT-cores. Dimension and rank information is inferred."""

    ml = loadmat(fname, simplify_cells=True)
    dU = ml["dU"]
    U = ml["U"]
    V = ml["V"]
    fix_shape(dU)
    fix_shape(U)
    fix_shape(V)

    tv = TensorTrainTangentVector(dU, U, V)
    if backend != "numpy":
        tv.convert_backend(backend)
    return tv


def save_matlab_ttmlv(ttmlv, fname):
    U = ttmlv.left_cores
    V = ttmlv.right_cores
    dU = ttmlv.grad_cores
    if ttmlv.backend != "numpy":
        U = convert_backend_cores(U, "numpy")
        V = convert_backend_cores(V, "numpy")
        dU = convert_backend_cores(dU, "numpy")
    savemat(
        fname,
        {
            "U": np.array(U, dtype="object"),
            "V": np.array(V, dtype="object"),
            "dU": np.array(dU, dtype="object"),
        },
    )


def ttmlv_to_matlab(ttmlv, eng):
    """Load TensorTrain into matlab engine as TTeMPS"""

    _, f = tempfile.mkstemp(suffix=".mat")
    save_matlab_ttmlv(ttmlv, f)
    U = eng.Py_to_TTeMPS_tangent_orth(f)
    os.remove(f)
    return U


def matlab_to_ttmlv(ttemps_tangent, eng, backend="numpy"):
    """Load matlab.object representing TTeMPS into Python as TT"""

    _, f = tempfile.mkstemp(suffix=".mat")
    eng.TTeMPS_tangent_orth_to_Py(f, ttemps_tangent, nargout=0)
    tt = load_matlab_ttmlv(f, backend)

    return tt
