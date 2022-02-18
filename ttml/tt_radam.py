"""Implements Riemannian ADAM stochastic gradient descent algorithm for tensor
trains. 

This is an implementation of the algorithm in the paper 'Riemannian Adaptive
Optimization Methods' by Becigneul and Ganea"""

import autoray as ar
import numpy as np

from .tt_opt import TensorTrainOptimizer
from .utils import merge_sum, convert_backend


class TensorTrainSGD(TensorTrainOptimizer):
    """
    Riemannian Adam optimizer for tensor trains.

    Parameters
    ----------
    tt : TensorTrain
        TensorTrain to be optimized. During optimization it will be copied, not
        modified.
    y : array<float64>
        Target values. Should be flat array with same backend as tt.
    idx : array<int64> shape `(len(y),tt.order)`
        Indices of dense tensor corresponding to values `y`. Potential
        duplicate values in `idx` are automatically merged.
    batch_size : int
    lr : float (default: 1.0)
        Learning rate, needs to be tuned for each problem.
    beta1 : float (default: 0.9)
        parameter between 0 and 1 determining the contribution of transport of
        previous search direction to current search direction
    beta2 : float (default: 0.9)
        parameter between 0 and 1 determining contribution of previous gradient
        norms to the stepsize.
    task : str (default: `"regression"`)
        Whether to perform regression or binary classification.

        - If `task="regression` then MSE is minimized.
        - If `task="classification"`. The labels are assumed to be 0 or 1, and
          cross entropy is minimized. Note that predictions of the classifier
          will be on the logit scale, only the objective changes.
    sample_weight : array<float64> or None (default: None)
        Weights associated to all sample points. If None, use unit weight.
    """
    def __init__(
        self,
        tt,
        y,
        idx,
        batch_size,
        lr=1.0,
        beta1=0.9,
        beta2=0.9,
        task="regression",
        sample_weight=None,
        red_idx=None,
        **kwargs
    ):
        super().__init__(
            tt,
            y,
            idx,
            task=task,
            sample_weight=sample_weight,
            red_idx=red_idx,
            **kwargs
        )
        self.batch_size = batch_size
        self.N = len(y)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = None
        self.adaptive_term = None

    def step(self):
        "Do one step inplace"
        batch_indices = np.random.randint(0, self.N, self.batch_size)
        batch_indices = convert_backend(batch_indices, self.backend)
        if self.backend == "tensorflow":
            batch_y = ar.do("take", self.y, batch_indices)
            batch_idx = ar.do("take", self.idx, batch_indices)
        else:
            batch_y = self.y[batch_indices]
            batch_idx = self.idx[batch_indices]
        loss, egrad = self.egrad(
            y=batch_y, idx=batch_idx, merge=False, normalize=True
        )
        batch_red_idx, egrad = merge_sum(batch_idx, egrad)
        rgrad = self.tt.rgrad_sparse(egrad, batch_red_idx)
        if self.momentum is None:
            self.momentum = rgrad
        else:
            self.momentum = self.tt.grad_proj(self.momentum)
            self.momentum = (
                self.beta1 * self.momentum + (1 - self.beta1) * rgrad
            )
        if self.adaptive_term is None:
            self.adaptive_term = rgrad.norm() ** 2
        else:
            self.adaptive_term = (
                self.beta2 * self.adaptive_term
                + (1 - self.beta2) * rgrad.norm() ** 2
            )
        step_size = self.lr / ar.do("sqrt", self.adaptive_term)
        self.tt.apply_grad(self.momentum, alpha=-step_size, inplace=True)
        derivative = -rgrad @ self.momentum
        return loss, derivative, step_size
