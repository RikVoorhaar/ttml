"""Linesearch for performing tensor completion with TensorTrains"""

import warnings
import autoray as ar
from autoray import numpy as np

from scipy.optimize import minimize
from scipy.optimize.linesearch import (
    scalar_search_armijo,
    scalar_search_wolfe2,
)

from ttml.tt_opt import TensorTrainOptimizer


class TensorTrainLineSearch(TensorTrainOptimizer):
    """Implements Riemannian conjugate gradient descent with linesearch for
    tensor train.

    Parameters
    ----------
    tt : TensorTrain
        TensorTrain to be optimized. During optimization it will be copied,
        not modified.
    y : array<float64>
        Target values. Should be flat array with same backend as tt.
    idx : array<int64> shape `(len(y),tt.order)`
        Indices of dense tensor corresponding to values `y`. Potential
        duplicate values in `idx` are automatically merged.
    task : str (default: `"regression"`)
        Whether to perform regression or binary classification.

        * If `task="regression` then MSE is minimized.
        * If `task="classification"`. The labels are assumed to be 0 or 1,
          and cross entropy is minimized. Note that predictions of the
          classifier will be on the logit scale, only the objective
          changes.
    sample_weight : array<float64> or None (default: None)
        Weights associated to all sample points. If None, use unit weight.
    cg_method : str (default: `"fr"`)
        Which conjugate gradient method to use. Currently supported are
        `'fr'` (Fletcher-Reeves), `'sd'` (steepest descent).
    line_search_method : str (default: `"armijo"`)
        Which line search method to use. Supported are `"armijo"`, `"wolfe"`
        (strong Wolfe conditions) and `"tnc"` (exact line search using TNC,
        mainly for debugging.)
    line_search_params : None or dict (default None)
        Extra kwargs to pass to the line search method.
    memory : int (default: 1)
        With memory > 1, perform nonmonotone line search with `memory` steps
        of memory. Using memory > 1 with Wolfe line search is not properly
        supported.
    max_stepsize : int or None (default: None)
        Maximum stepsize to be taken.
    initial_stepsize_method : str (default: "bb1")
        Method to compute initial stepsize for backtracking. Allowed values:

        * "bb1" : Riemannian Barzilai-Borwein stepsize of first type.
        * "bb2" : Riemannian Barzilai-Borwein stepsize of second type.
        * "qopt" : Quasi-optimal Riemannian CG stepsize, as proposed by
          Steinlechner.
        * "scalar" : take twice the difference between current and previous loss
          value, divided by the derivative of the line search function.
    last_step_size : int or None
        Last step size taken, to be used for a warm start of the optimizer.
    default_stepsize : float (default: 1.0)
        Fall back default stepsize
    auto_scale : bool (default False)
        Use the first gradient norms to estimate the scale of the step size.
        Only affects first step size. This is useful if optimal stepsize is 
        particularly large.
    """

    def __init__(
        self,
        tt,
        y,
        idx,
        task="regression",
        sample_weight=None,
        red_idx=None,
        cg_method="fr",
        line_search_method="armijo",
        line_search_params=None,
        memory=1,
        max_stepsize=None,
        min_initial_stepsize=1e-6,
        initial_stepsize_method="bb1",
        last_step_size=None,
        default_stepsize=None,
        auto_scale=False,
        **kwargs,
    ):
        super().__init__(
            tt,
            y,
            idx,
            task=task,
            sample_weight=sample_weight,
            red_idx=red_idx,
            **kwargs,
        )
        self.cg_method = cg_method
        if cg_method == "fr":
            self.cg_beta = self.cg_fletcher_reeves
        elif cg_method == "sd":
            self.cg_beta = self.cg_constant
        else:
            raise ValueError(
                f"Unsupported CG method '{cg_method}', choose from 'fr','sd'"
            )

        self.line_search_method_name = line_search_method
        if line_search_method == "armijo":
            self.line_search_method = armijo_backtracking
        elif line_search_method == "wolfe":
            self.line_search_method = strong_wolfe_line_search
        else:
            raise ValueError(
                f"""Unsupported linesearch method '{line_search_method}', use
                'armijo' or 'wolfe"""
            )
        if line_search_params is None:
            self.line_search_params = dict()
        else:
            self.line_search_params = line_search_params

        self.memory = memory
        self.max_stepsize = max_stepsize
        self.min_initial_stepsize = min_initial_stepsize
        self.initial_stepsize_method = initial_stepsize_method

        self.new_tt = None
        self.old_phi0 = None
        self.old_derphi0 = None
        self.last_step_size = last_step_size
        if default_stepsize is None and last_step_size is not None:
            self.default_stepsize = last_step_size
        else:
            self.default_stepsize = 1.0
        self._need_autoscale = auto_scale
        self._last_step = None
        self.prev_loss = None
        self._step_size_dic = dict()

        self.loss_current = None
        self.loss_prev = None
        self.rgrad = None
        self.rgrad_transp = None
        self.egrad_current = None
        self.egrad_prev = None
        self.rsearch_dir = None
        self.esearch_dir = None
        self.rsearch_dir_transp = None
        self.esearch_dir_transp = None

        self._derphi_cache = dict()
        self._new_tt_phi = None

    def _init_loss(self, try_sd=False):
        self._derphi_cache = dict()

        self.egrad_prev = self.egrad_current
        self.loss_prev = self.loss_current
        loss, self.egrad_current = self.egrad()
        if self.rsearch_dir is not None:
            self.rsearch_dir_transp = self.tt.grad_proj(self.rsearch_dir)
            self.esearch_dir_transp = self.rsearch_dir_transp.to_eucl(
                self.red_idx
            )
        if self.rgrad is not None:  # TODO: Only compute this if we need it
            self.rgrad_transp = self.tt.grad_proj(self.rgrad)
        self.rgrad = self.tt.rgrad_sparse(self.egrad_current, self.red_idx)
        if self.rsearch_dir_transp is not None:
            if try_sd:
                self.rsearch_dir = -self.rgrad
            else:
                beta = self.cg_beta()
                self.rsearch_dir = -self.rgrad + self.rsearch_dir_transp * beta
        else:
            self.rsearch_dir = -self.rgrad

        derphi0 = self.rgrad @ self.rsearch_dir
        return loss, derphi0

    def _phi_derphi(self, alpha, riemannian=False):
        """Do step of size alpha in search direction and compute gradient,
        transported gradient, and if doing CG, the transported search direction

        If `riemannian=True` use Riemannian gradient for loss derivative. Should
        not be different.
        """
        new_tt = self.tt.apply_grad(
            self.rsearch_dir, alpha=alpha, round=True, inplace=False
        )
        phi, new_egrad = self.egrad(new_tt)
        rsearch_transpr = new_tt.grad_proj(self.rsearch_dir)
        if riemannian:
            new_rgrad = new_tt.rgrad_sparse(new_egrad, self.red_idx)
            derphi = rsearch_transpr @ new_rgrad
        else:
            esearch_transpr = rsearch_transpr.to_eucl(self.red_idx)
            derphi = ar.do("dot", esearch_transpr, new_egrad)
        self._new_tt_phi = (new_tt, phi)
        return phi, derphi

    def _phi(self, alpha):
        import numpy

        try:
            new_tt = self.tt.apply_grad(
                self.rsearch_dir, alpha=alpha, round=True, inplace=False
            )
        except numpy.linalg.LinAlgError:
            # LinAlgError points to converged line search.
            # This exception is handled in TTML.fit()
            raise
        phi = self.loss(tt=new_tt, normalize=False)
        self._new_tt_phi = (new_tt, phi)
        return phi

    def _phi_with_grad(self, alpha):
        # scipy.optimize feeds 1-element arrays that need to converted to scalar
        if not ar.infer_backend(alpha) == "builtins":
            alpha = float(ar.reshape(alpha, (-1,))[0])
        phi, derphi = self._phi_derphi(alpha)
        self._derphi_cache[alpha] = derphi
        return phi

    def _derphi(self, alpha):
        # scipy.optimize feeds 1-element arrays that need to converted to scalar
        if not isinstance(alpha, float):
            alpha = ar.to_numpy(alpha).reshape(-1)[0]
        if alpha not in self._derphi_cache:
            _, derphi = self._phi_derphi(alpha)
            self._derphi_cache[alpha] = derphi
        return self._derphi_cache[alpha]

    def quasi_optimal_stepsize(self, derphi0, default_stepsize=1.0):
        r"""Compute the quasi-optimal stepsize based on a linearization.

        The formula for this is

        .. math ::
            -\phi'(0) / \|\eta\|^2

        With :math:`\eta` the search direction, 
        :math:`\phi'(0) = \langle\eta,\nabla f\rangle` and
        the derivative of the line search objective

        Returns
        -------
        step_size : float
        """
        numerator = -derphi0
        denominator = self.rsearch_dir.norm() ** 2
        if denominator != 0 and numerator > 0:
            step_size = numerator / denominator
        else:
            step_size = default_stepsize
        return step_size

    def scalar_stepsize(self, phi0, old_phi0, derphi0):
        if self.last_step_size is None:
            old_step_size = 1.0
        else:
            old_step_size = self.last_step_size
        if old_phi0 is not None and derphi0 != 0:
            alpha0 = 1.01 * 2 * (phi0 - old_phi0) / derphi0
        else:
            alpha0 = old_step_size
        return alpha0

    def bb_stepsize(self, bb_type=1):
        """Riemannian Barzilai-Borwein stepsize.

        There are two variants of the BB stepsize, this is controlled by the
        argument `bb_type`. If the stepsize cannot be computed, or would be
        excessively high, `default_stepsize` is returned instead."""
        if self.rgrad_transp is None:
            return self.default_stepsize
        S = self.rgrad_transp * self.last_step_size
        Y = self.rgrad - self.rgrad_transp
        SY = np.abs(S @ Y)
        if bb_type == 1:
            s_norm_squared = S.norm() ** 2
            if (
                SY < 1e-4 * s_norm_squared / self.last_step_size
            ):  # If S and Y are almost orthogonal
                return self.default_stepsize
            else:
                return s_norm_squared / SY
        else:  # bb_type = 2
            Y_norm_squared = Y.norm() ** 2
            if np.abs(Y_norm_squared) < 1e-4 * SY / self.last_step_size:
                return self.default_stepsize
            else:
                return SY / Y_norm_squared

    def step(self, try_armijo=False, try_sd=False):
        """Perform a step. Replaces self.tt updated tt.

        Parameters
        ----------
        try_armijo : bool, default=False
            Force using armijo linesearch. This is called if wolfe line search
            fails
        try_sd : bool, default=False
            If armijo also fails, it may be because the search direction is bad,
            so we use this to force steepest descent direction.

        Returns
        -------
        phi0 : float
            New value of the loss function
        derphi0 : float
            Derivative of loss function in search direction (at beginning of
            step)
        step_size : float
            Size of step taken
        """
        phi0, derphi0 = self._init_loss(try_sd=try_sd)
        if not (try_sd or try_armijo):  # avoid adding the loss value twice
            self.loss_history.append(phi0)
        if self._need_autoscale:  # Ugly hack for tiny gradients
            self.default_stepsize = -1 / derphi0
            self.last_step_size = -1 / derphi0
            self._need_autoscale = False

        phi_information = {
            "phi0": np.max(self.loss_history[-self.memory :]),
            "derphi0": derphi0,
            "old_phi0": self.old_phi0,
            "old_derphi0": self.old_derphi0,
            "old_step_size": self.last_step_size,
            "phi_history": self.loss_history,
        }
        if try_armijo:
            line_search_method = armijo_backtracking
        else:
            line_search_method = self.line_search_method

        if self.initial_stepsize_method == "bb1":
            alpha0 = self.bb_stepsize(bb_type=1)
        elif self.initial_stepsize_method == "bb2":
            alpha0 = self.bb_stepsize(bb_type=2)
        elif self.initial_stepsize_method == "qopt":
            step_size = self.quasi_optimal_stepsize(derphi0)
            alpha0 = step_size
        elif self.initial_stepsize_method == "scalar":
            alpha0 = self.scalar_stepsize(phi0, self.old_phi0, derphi0)
        else:
            raise ValueError(
                f"Unknown init stepsize method {self.initial_stepsize_method}"
            )
        if self.min_initial_stepsize is not None:
            alpha0 = max(alpha0, self.min_initial_stepsize)
        if self.max_stepsize is not None:
            alpha0 = min(self.max_stepsize, alpha0)

        if self.line_search_method_name == "wolfe" and not try_armijo:
            phi_func = self._phi_with_grad
        else:
            phi_func = self._phi
        step_size = line_search_method(
            phi_func,
            self._derphi,
            alpha0=alpha0,
            amax=self.max_stepsize,
            **phi_information,
            **self.line_search_params,
        )

        if step_size is not None:
            self.old_phi0 = phi0
            self.old_derphi0 = derphi0
            # self.tt = self.tt.apply_grad(
            #     self.rsearch_dir, alpha=step_size, round=True, inplace=False
            # )
            # phi0 = self.loss_func()
            self.tt, phi0 = self._new_tt_phi
            phi0 /= len(self.y)

            self.step_size_history.append(step_size)
            self.last_step_size = step_size
            return phi0, derphi0, step_size
        else:  # Use fallback methods in case we didn't find a suitable stepsize
            if not try_armijo and self.line_search_method_name == "wolfe":
                return self.step(try_armijo=True, try_sd=try_sd)
            elif not try_sd and self.cg_method != "sd":
                return self.step(
                    try_armijo=try_armijo,
                    try_sd=True,
                )
            # If fallback fails, take same step as last time and raise warning
            else:  
                self.tt = self.tt.apply_grad(
                    self.rsearch_dir,
                    alpha=self.last_step_size,
                    round=True,
                    inplace=False,
                )
                self.step_size_history.append(self.last_step_size)
                phi0 = self.loss()
                warnings.warn(
                    """Linesearch failed to converge. Probably convergence has
                    been reached.""",
                    RuntimeWarning,
                )
                return phi0, derphi0, step_size

    def plot_linesearch(
        self, alpha0=1.0, alpha_max=None, c1=1.3, c2=0.5, plot_points=20
    ):
        """Return arrays to plot linesearch objective for debugging.

        Parameters
        ----------
        alpha0: initial point
        alpha_max : float or None
            If specified, skip finding a good alpha
        c1: factor to increase alpha by every step if derivative negative
        c2: factor to decrease alpha by every step if derivative positive
        plot_points: number of points returned

        Returns
        -------
        plot_X: X positions of plot points
        phis: values of objective at plot_X
        der_phis: derivative of objective at plot_X
        """

        phi0, derphi0 = self._init_loss()
        if alpha_max is None:
            alpha = alpha0
            alphas = [0]
            derphis = [derphi0]
            phis = [phi0]

            # Find a good alpha
            for _ in range(100):
                phi, derphi = self._phi_derphi(alpha)
                alphas.append(alpha)
                phis.append(phi)
                derphis.append(derphi)
                if phi > phi0:
                    alpha = min(alphas[1:]) * c2
                elif derphi < 0:
                    alpha *= c1
                else:
                    break

            plot_X = np.linspace(
                0, alphas[np.argmin(phis)] * c1 ** 2, plot_points
            )
        else:
            plot_X = np.linspace(0, alpha_max, plot_points)
        phis, derphis = np.array([self._phi_derphi(x) for x in plot_X]).T

        return plot_X, phis, derphis

    def cg_constant(self):
        """Constant scale parameter. If alpha=0, then this is steepest
        descent."""

        return 0

    def cg_fletcher_reeves(self):
        r"""Fletcher-Reeves scale parameter.

        This is given by

        .. math::
            \beta_{k+1}^{FR} = \frac{\langle\nabla f(x_{k+1}),\,
            \nabla f(x_{k+1})\rangle_{x_{k+1}}}
            {\langle\nabla f(x_k),\nabla f(x_k)\rangle_{x_k}}
        """
        numerator = ar.do("dot", self.egrad_current, self.egrad_current)
        denominator = ar.do("dot", self.egrad_prev, self.egrad_prev)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator


TTLS = TensorTrainLineSearch


################################################################################
# Line search methods
################################################################################


def strong_wolfe_line_search(
    phi,
    derphi,
    phi0=None,
    old_phi0=None,
    derphi0=None,
    c1=1e-4,
    c2=0.9,
    amax=None,
    **kwargs,
):
    """
    Scalar line search method to find step size satisfying strong Wolfe
    conditions.

    Parameters
    ----------
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size

    Returns
    -------
    step_size : float
        The next step size
    """

    step_size, _, _, _ = scalar_search_wolfe2(
        phi,
        derphi,
        phi0=phi0,
        old_phi0=old_phi0,
        c1=c1,
        c2=c2,
        amax=amax,
    )

    return step_size


def armijo_backtracking(
    phi,
    derphi,
    phi0=None,
    derphi0=None,
    old_phi0=None,
    c1=1e-4,
    amin=0,
    amax=None,
    old_step_size=1.0,
    alpha0=None,
    **kwargs,
):
    """Scalar line search method to find step size satisfying Armijo conditions.

    Parameters
    ----------
    c1 : float, optional
        Parameter for Armijo condition rule.
    amax, amin : float, optional
        Maxmimum and minimum step size
    """
    if alpha0 is None:
        if old_step_size is None:
            old_step_size = 1.0
        if old_phi0 is not None and derphi0 != 0:
            alpha0 = 1.01 * 2 * (phi0 - old_phi0) / derphi0
        else:
            alpha0 = old_step_size
    if alpha0 <= 0:
        alpha0 = old_step_size
    if amax is not None:
        alpha0 = min(alpha0, amax)

    step_size, _ = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=amin
    )

    return step_size


def nonmonotone_armijo(
    phi,
    derphi,
    alpha0=None,
    derphi0=None,
    phi_history=None,
    memory=5,
    c1=1e-4,
    sigma=0.7,
    amin=0,
    **kwargs,
):
    if alpha0 is None:
        alpha = 1.0
    else:
        alpha = alpha0
    if derphi0 is None:
        derphi0 = derphi(0)
    if len(phi_history) == 0:
        max_hist = phi(0)
    else:
        max_hist = np.max(phi_history[-memory:])

    while alpha > amin:
        new_phi = phi(alpha)
        if new_phi <= max_hist + c1 * alpha * derphi0:
            break
        alpha *= sigma
    return alpha


def TNC_exact_linesearch(phi, derphi, old_step_size=None, **kwargs):
    if old_step_size is None:
        old_step_size = 1.0
    res = minimize(phi, old_step_size, jac=derphi, method="TNC")
    return res.x[0]
