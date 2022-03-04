"""Meta optimizer class for Tensor Trains"""


import autoray as ar
from ttml.utils import merge_sum, convert_backend


class TensorTrainOptimizer:
    """
    Parameters
    ----------
    tt : TensorTrain
        TensorTrain to be optimized. During optimization it will be copied, not
        modified.
    y : array<float64>
        Target values. Should be flat array with same backend as `tt`.
    idx : array<int64> shape `(len(y),tt.order)`
        Indices of dense tensor corresponding to values `y`. Potential
        duplicate values in `idx` are automatically merged.
    task : str (default: `"regression"`)
        Whether to perform regression or binary classification.

        * If `task="regression` then MSE is minimized.

        * If `task="classification"`. The labels are assumed to be 0 or 1, and
          cross entropy is minimized. Note that predictions of the classifier
          will be on the logit scale, only the objective changes.
    sample_weight : array<float64> or None (default: None)
        Weights associated to all sample points. If None, use unit weight.
    red_idx : array<int, int>  or None
        Unique indices of `idx` in lexicographic order. If `None` (default) this
        is computed.
    task_kwargs : dict (optional)
        Dictionary of keyword arguments to always be passed to the loss and
        egrad functions.
    """

    def __init__(
        self,
        tt,
        y,
        idx,
        task="regression",
        sample_weight=None,
        red_idx=None,
        task_kwargs=None,
        **kwargs
    ):
        self.tt = tt
        self.idx = idx
        self.y = y
        self.backend = tt.backend
        self.task = task
        self._loss_func = _loss_func_dict[task]
        self._egrad_func = _egrad_func_dict[task]
        self.sample_weight = sample_weight
        if task_kwargs is None:
            task_kwargs = dict()
        self.task_kwargs = task_kwargs
        if red_idx is None:
            red_idx, _ = merge_sum(
                idx,
                ar.do(
                    "zeros",
                    (len(y),),
                    like=self.backend,
                    dtype=ar.to_backend_dtype("float64", self.backend),
                ),
            )
        self.red_idx = red_idx

        self.loss_history = []
        self.step_size_history = []
        self.func_calls = 0
        self.grad_calls = 0

    def step(self):
        """Do a step.

        This should return loss at new point, inner product between
        search-direction and gradient, and step size of new step.

        Should be implemented by the inhereting class."""
        raise NotImplementedError

    def loss(
        self,
        tt=None,
        y=None,
        idx=None,
        sample_weight=None,
        normalize=True,
        **kwargs
    ):
        """Compute the loss at current point.

        Returns
        -------
        loss : float

        Parameters
        ----------
        tt : TensorTrain or None
            Compute loss at `tt` instead. If `None` (default) then compute at
            `self.tt`.
        y : array<float> or None
            Target labels for loss. If `None` (default), use `self.y`
        idx : array<int, int>  or None
            Tensor indices to use for loss. If `None` (default) use `self.idx`
        sample_weight : array<float> or None
            Array same shape as `y` giving sample weights. If `None`, use weight
            1 for each entry
        normalize : bool (default: True)
            Divide loss function by number of samples (or sum of
            `sample_weight`) if True
        """
        self.func_calls += 1
        if tt is None:
            tt = self.tt
        if y is None:
            y = self.y
        if idx is None:
            idx = self.idx
        if sample_weight is None:
            sample_weight = self.sample_weight
        return self._loss_func(
            tt=tt,
            y=y,
            idx=idx,
            sample_weight=sample_weight,
            normalize=normalize,
            **self.task_kwargs,
            **kwargs,
        )

    def egrad(
        self,
        tt=None,
        y=None,
        idx=None,
        sample_weight=None,
        normalize=False,
        merge=True,
        **kwargs
    ):
        """Compute the sparse Euclidean gradient and loss at current point.

        Keyword arguments are the same as `self.loss`, except `normalize=False`
        by default.

        Returns
        -------
        loss : float
        egrad : array<float>
            Array with length the number of unique entries in `idx`. Corresponds
            to sparse Euclidean tangent vector, with indices obtained by a
            lexical sort applied to `idx`. See also :meth:`utils.merge_sum`.
        """
        self.grad_calls += 1
        self.func_calls += 1
        if tt is None:
            tt = self.tt
        if y is None:
            y = self.y
        if idx is None:
            idx = self.idx
        if sample_weight is None:
            sample_weight = self.sample_weight
        loss, egrad = self._egrad_func(
            tt=tt,
            y=y,
            idx=idx,
            sample_weight=sample_weight,
            normalize=normalize,
            **self.task_kwargs,
            **kwargs,
        )
        if merge:
            _, egrad = merge_sum(idx, egrad)
        return loss, egrad


def _regression_loss(tt, y, idx, sample_weight=None, normalize=True, **kwargs):
    residuals = (tt.gather(idx) - y) ** 2
    if sample_weight is not None:
        residuals *= sample_weight
    if normalize:
        return ar.do("mean", residuals)
    else:
        return ar.do("sum", residuals)


def _classification_loss(
    tt, y, idx, sample_weight=None, normalize=True, **kwargs
):
    p = ar.do("sigmoid", tt.gather(idx))
    p = convert_backend(p, ar.infer_backend(y))
    one_min_p = 1 - p
    p = ar.do("clip", p, 1e-8, 1)
    one_min_p = ar.do("clip", one_min_p, 1e-8, 1)
    cross_entropy = -y * ar.do("log", p) - (1 - y) * ar.do("log", one_min_p)
    if sample_weight is not None:
        cross_entropy *= sample_weight
    if normalize:
        return ar.do("mean", cross_entropy)
    else:
        return ar.do("sum", cross_entropy)


_loss_func_dict = {
    "regression": _regression_loss,
    "classification": _classification_loss,
}


def _regression_egrad(tt, y, idx, sample_weight=None, normalize=True, **kwargs):
    residuals = tt.gather(idx) - y
    residuals_wt = residuals
    if sample_weight is not None:
        residuals_wt *= sample_weight
    loss = ar.do("dot", residuals, residuals_wt)
    egrad = 2 * residuals_wt
    if normalize:
        loss /= len(y)
        egrad /= len(y)
    return loss, egrad


def _classification_egrad(
    tt, y, idx, sample_weight=None, normalize=True, **kwargs
):
    p = ar.do("sigmoid", tt.gather(idx))
    p = convert_backend(p, ar.infer_backend(y))
    one_min_p = 1 - p
    p = ar.do("clip", p, 1e-8, 1)
    one_min_p = ar.do("clip", one_min_p, 1e-8, 1)
    cross_entropy = -y * ar.do("log", p) - (1 - y) * ar.do("log", one_min_p)
    if sample_weight is not None:
        cross_entropy *= sample_weight
    cross_entropy = ar.do("sum", cross_entropy)
    loss = cross_entropy
    residuals = p - y
    if sample_weight is not None:
        residuals *= sample_weight
    egrad = residuals
    if normalize:
        loss /= len(y)
        egrad /= len(y)
    return loss, egrad


_egrad_func_dict = {
    "regression": _regression_egrad,
    "classification": _classification_egrad,
}
