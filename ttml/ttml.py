"""
Implements TTML, a tensor train based machine learning estimator.

Uses existing machine learning estimators to initialize a tensor train
decomposition on a particular feature space discretization.

Then this tensor train is further optimized with Riemannian conjugate gradient
descent. This library also implements much functionality related to tensor
trains. And their Riemannian optimization in general.

One can use :class:`TTMLRegressor` and :class:`TTMLClassifier` just like any
`scikit-learn` estimator. As parameter it just needs another `scikit-learn`-like
estimator. For example, here we fit a :class:`TTMLRegressor` using
:class:`RandomForestRegressor` for initialization.

>>> from ttml import ttml
... import numpy as np
... from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
...
... # Try to learn the summation function in 10 dimensions
... X = np.random.normal(size=(1000, 10))
... y = np.sum(X, axis=1)
...
... forest = RandomForestRegressor()
... ttml = ttml.TTMLRegressor(forest)
... ttml.fit(X, y)
TTMLRegressor(estimator=RandomForestRegressor())

Note that here there is no need to fit random forest to data separately; this is
done automatically when calling :meth:`ttml.fit()` (but only if the random forest
has not been fitted to data yet). After fitting, we can use :meth:`.predict()`
for predicting values:

>>> ttml.predict([[0.3] * 10]) # ten times the same value
array([2.89993562])

For classification problems only data with 0/1 labels is supported, but 
otherwise the procedure is very similar to regression. For example below
we try to learn the function which returns 1 only if the sum of 10 numbers
is positive.

>>> X = np.random.normal(size=(1000, 10))
... y = (np.sum(X, axis=1) > 0).astype(int)
... 
... forest = RandomForestClassifier()
... ttml = ttml.TTMLClassifier(forest)
... ttml.fit(X, y)
TTMLClassifier(estimator=RandomForestClassifier())

For prediction, we can use :meth:`.predict()` which gives 0/1 labels as output,
:meth:`.predict_proba()` which gives a probability, and :meth:`predict_logit()` 
which outputs a logit.

>>> ttml.predict([[0.3] * 10])
array([1.])

>>> ttml.predict_proba([[0.3] * 10])
array([0.9999665])

>>> ttml.predict_logit([0.3] * 10)
array([10.30384462])

Other than the base estimator, the most important hyperparameters for the
ttml are the `tensor train rank` and `number of thresholds per feature`,
controlled by the keywords ``max_rank`` and ``num_thresholds`` respectively. The
respective default values are ``5`` and ``50``. Changing the ``max_rank``
parameter can cause fitting and inference (prediction) to take significantly
longer. The ``num_thresholds`` parameter also affects fittings speed, but does
not affect inference speed. Both parameters affect the accuracy of the model,
and there is no general rule of thumb for the optimal value of these parameters.

>>> ttml = ttml.TTMLClassifier(forest, max_rank=2, num_thresholds=10)
... ttml.fit(X, y)
TTMLClassifier(estimator=RandomForestClassifier())

Another feature that can greatly improve performance of ttml, is early
stopping. During the Riemannian optimization phase, we can monitor the
performance on a validation dataset. This greatly reduces the tendency to
overfit. To do this, we simply need to specify a validation dataset to the
:meth:`fit` method:

>>> from sklearn.model_selection import train_test_split
... X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
... ttml.fit(X_train, y_train, X_val=X_val, y_val=y_val)
TTMLClassifier(estimator=RandomForestClassifier())
"""

import numbers
from datetime import datetime
import warnings
from time import perf_counter_ns

import autoray as ar
import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.tree._classes import BaseDecisionTree
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


from .tensor_train import TensorTrain
from .tt_rlinesearch import TTLS, TensorTrainLineSearch
from .forest_compression import compress_forest_thresholds
from .tt_cross import estimator_to_tt_cross
from .utils import convert_backend, predict_logit

_fit_parameters_string = """

Parameters
----------
X : np.ndarray
    Training X data values
y : np.ndarray
    Training truth labels. Should be 0/1 for classification.
estimator :
    An `sklearn`-like estimator for fitting. If the estimator is not yet
    fitted, this calls the estimator's .fit function. For classification
    tasks, take note of the `estimator_output` keyword. If your
    estimator does not have an appropriate .predict method, then pass
    _predict_fun as a last resort. For classification we assume this
    outputs logits.
X_val : np.ndarray (optional)
    Validation set to monitor during training
y_val: np.ndarray (optional)
task : str (default: "regression")
    Whether to use "regression" or "classification" as task. In the case
    of "classification" this estimator will output logits.
num_thresholds : int (default: 50)
    The number of thresholds per feature to pick from the forest.
    Ignored if the `_thresholds` kwarg is specified.
tt_cross_its : int (default: 5)
    Number of iterations for the TT-cross algorithm
max_rank : int (default: 10)
    Maximum rank for the tensor train
opt_steps : int (default: 100)
    Number of steps of Riemannian conjugate gradient descent to take
opt_tol : float (default:  1e-5)
    After 3 steps of no relative improvement of at least `opt_tol`, the
    Riemannian conjugate gradient descent is stopped. If `X_val` is
    supplied then error is monitored on validation set, otherwise on
    training set.
estimator_output : str (default: "logit")
    For classification tasks, the output of the estimators' .predict
    function. Supported arguments are "logit" and "proba". If the
    estimator has a .predict_proba method, then this is ignored and
    .predict_proba is used instead.
verbose : bool (default: False)
    If True, print convergence and debug information
_thresholds : list[np.ndarray] (optional)
    Use this list of thresholds instead of inferring from the forest.
    Should be a list of arrays, one array per feature. The last element
    of each array is expected to be `np.inf`. TODO: update
_ttls_kwargs : dict (optional)
    Keyword arguments to pass to the Riemannian conjugate gradient
    descent optimizer. See `TensorTrainLineSearch` for details.
_predict_fun : method (optional)
"""


class TTML:
    """Implements a TTML.

    This stores a TensorTrain and a list of thresholds for each feature. Can be
    used as an sklearn-like estimator once trained.

    Not intended to be initialized directly for most use cases, use
    :class:`TTMLClassifier` and :class:`TTMLRegressor` instead.

    Parameters
    ----------
    tt : TensorTrain
    thresholds : list<np.ndarray>
    categorical_features : tuple<int> or None
    """

    def __init__(self, tt, thresholds, categorical_features=None):
        if categorical_features is None:
            self.categorical_features = tuple()
        else:
            self.categorical_features = categorical_features

        self.tt = tt
        self.thresholds = thresholds
        self.n_features = len(thresholds)
        self.backend = tt.backend

    @property
    def num_params(self):
        n_thresh_params = sum(np.size(t) for t in self.thresholds)
        n_tt_params = self.tt.num_params()
        return n_thresh_params + n_tt_params

    @classmethod
    def from_tree(cls, decision_tree):
        """Initialize from sklearn decision tree.

        This creates a lossless encoding of the decision tree as TTML.
        The thresholds are precisely the decision boundaries of the tree.
        """

        # Support both decision trees and sklearn.tree._tree.Tree objects.
        if isinstance(decision_tree, BaseDecisionTree):
            decision_tree = decision_tree.tree_
        thresholds, leaf_values, leaf_filter_matrices = cls._tree_to_tensor(
            decision_tree
        )
        tt = cls.tree_to_tt(thresholds, leaf_values, leaf_filter_matrices)
        return cls(tt, thresholds)

    @staticmethod
    def _tree_to_tensor(tree, thresholds=None):
        """Return tensor encoding tree in CP format together with thresholds.

        Parameters
        ----------
        tree : :meth:`sklearn.tree.tree_.Tree` currently only works with tree
            coming from DecisionTreeRegressor

        thresholds : list[np.ndarray] Don't infer thresholds from tree, but use
            a precomputed list instead.

        Returns
        -------
        thresholds : list[np.ndarray] One array of thresholds for each feature.
            First value is always `-inf`

        leaf_values : np.ndarray Decision labels at leaves

        leaf_filter_matrices : list[np.ndarray[bool]] For each feature a matrix
            of shape (n_thresholds, n_leaves). Each row encodes the filter
            values for this particular feature and leaf.
        """

        if isinstance(tree, BaseDecisionTree):
            tree = tree.tree_

        # Compute thresholds for each feature
        feat_inds = [
            np.where(tree.feature == feat) for feat in range(tree.n_features)
        ]
        if thresholds is None:
            thresholds = []
            for inds in feat_inds:
                thresholds.append(
                    np.unique(
                        np.concatenate([[np.infty], tree.threshold[inds]])
                    )
                )

        # Decision labels of leaves
        leaves = np.where(tree.feature < 0)
        leaf_values = tree.value[leaves][:, 0, 0]

        # For each node, initialize filter matrix to be True for all thresholds
        filter_matrices = [
            np.full((tree.node_count, len(thresholds[i])), True)
            for i in range(tree.n_features)
        ]

        for index in range(tree.node_count):
            l_child = tree.children_left[index]
            r_child = tree.children_right[index]
            if l_child < 0:  # leaf node
                continue

            feat = tree.feature[index]
            threshold = tree.threshold[index]
            new_filter = thresholds[feat] <= threshold

            # Let children inherit filter matrices
            for f in range(tree.n_features):
                filter_matrices[f][l_child] = filter_matrices[f][index].copy()
                filter_matrices[f][r_child] = filter_matrices[f][index].copy()

            # Update filter matrices of children
            filter_matrices[feat][l_child] *= new_filter
            filter_matrices[feat][r_child] *= ~new_filter

        # Only store filters of leaves
        leaf_filter_matrices = [
            filt[leaves].astype(np.float64) for filt in filter_matrices
        ]
        return thresholds, leaf_values, leaf_filter_matrices

    @staticmethod
    def tree_to_tt(thresholds, leaf_values, leaf_filter_matrices):
        """Convert sklearn decision tree into a TT together with threshold
        arrays"""
        cores = []
        d = len(thresholds)
        for i, X in enumerate(leaf_filter_matrices):
            if i == 0:  # make shape (1,k[0],r)
                core = (leaf_values * X.T).reshape(1, X.shape[1], X.shape[0])
            elif i == d - 1:  # make shape (r,k[d-1],1)
                core = X.reshape(X.shape[0], X.shape[1], 1)
            else:  # make shape (r,k[i],r), diagonal along axis 0 and 2.
                core = np.apply_along_axis(np.diag, 0, X).transpose(0, 2, 1)
                # cores in the long run.
            cores.append(core)

        # Orthogonalize, reducing rank
        tt = TensorTrain(cores, is_orth=True)
        tt.orthogonalize(force_rank=False)

        return tt

    @staticmethod
    def thresholds_from_data(X, n_thresholds, categorical_features=None):
        """Compute thresholds from data by binning according to percentile.

        The last threshold is always `np.infty`. For categorical features all
        the unique values are used as thresholds, and the last value is replaced
        by `np.infty`.

        Parameters
        ----------
        X : np.ndarray
        n_thresholds : int or iterable<int>
            The number of thresholds to use for each feature. Ignored for
            categorical features.
        categorical_features : tuple (optional)
            The indices of the categorical features.

        Returns
        -------
        thresholds : list<np.ndarray>
        """
        if isinstance(n_thresholds, int):
            n_thresholds = [n_thresholds] * X.shape[1]
        if categorical_features is None:
            categorical_features = tuple()

        thresholds = []
        for i in range(X.shape[1]):
            if i in categorical_features:
                thresholds.append(np.unique(X[:, i]))
                thresholds[i][-1] = np.infty
            else:
                X_values = X[:, i][X[:, i] < np.max(X[:, i])]
                percents = np.arange(1, n_thresholds[i]) * 100 / n_thresholds[i]
                try:
                    thresholds.append(
                        np.percentile(
                            X_values, percents, interpolation="nearest"
                        )
                    )
                except IndexError:
                    # Handle index errors, apparently X_values is empty
                    thresholds.append(np.array([np.min(X[:, i])]))
                thresholds[i] = np.concatenate([thresholds[i], [np.infty]])
            thresholds[i] = np.unique(thresholds[i])
        return thresholds

    @classmethod
    def random_from_data(
        cls, X, rank, n_thresholds, backend="numpy", categorical_features=None
    ):
        """Make a TTML with random TT and with thresholds determined by
        :meth:`TTML.thresholds_from_data`

        Parameters
        ----------
        X : np.ndarray
        rank : int or iterable<int>
            The tensor-train rank. If a list, it should be of length one shorter
            than number of features
        n_thresholds : int or interable<int>
            Number of thresholds to use for each feature. This determines the
            outer dimensions of the TT
        backend : str, optional (default: 'numpy')
        categorical_features : tuple<int> or None
            The indices of the categorical features if any
        """

        thresholds = cls.thresholds_from_data(
            X, n_thresholds, categorical_features=categorical_features
        )

        tt = TensorTrain.random(
            [len(t) for t in thresholds], rank, backend=backend
        )

        return cls(tt, thresholds, categorical_features=categorical_features)

    def observed_indices(self, X):
        inds = np.array(
            [
                np.searchsorted(self.thresholds[i], X[:, i])
                for i in range(self.n_features)
            ]
        ).T

        return convert_backend(inds, self.backend)

    def predict(self, X, task="regression"):
        """Predict values for the TT `tt` using threshold arrays and input
        `X`"""
        # Look up location of X in threshold arrays (and subtract 1)
        inds = self.observed_indices(X)
        predictions = self.tt.gather(inds)
        predictions = ar.to_numpy(predictions)
        if task == "classification":
            predictions = predict_logit(predictions)
        return predictions

    def expand_thresholds(self, thresholds):
        """Add new tresholds to TT (inplace), merging duplicate thresholds.

        The TT-cores for the new thresholds are copied from those already
        present. The predictions of expanded model are guaranteed to be the
        same.
        """
        new_thresholds = []
        new_cores = []
        for i in range(len(self.thresholds)):
            # Do nothing with categorical features
            if i in self.categorical_features:
                new_thresholds.append(self.thresholds[i])
                new_cores.append(self.tt.cores[i])
                continue

            # Merge thresholds
            new_threshold = np.sort(
                np.unique(np.concatenate([self.thresholds[i], thresholds[i]]))
            )

            # New core is obtained by repeating slices in axis 1
            thresh_inds = np.searchsorted(self.thresholds[i], new_threshold)
            thresh_inds = np.clip(thresh_inds, 0, len(self.thresholds[i]) - 1)
            thresh_inds = convert_backend(thresh_inds, self.backend)
            core = self.tt.cores[i]
            new_core = ar.do("take", core, thresh_inds, axis=1)

            new_thresholds.append(new_threshold)
            new_cores.append(new_core)

        self.tt = TensorTrain(new_cores)
        self.tt.orthogonalize(force_rank=False)
        self.thresholds = new_thresholds

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            raise NotImplementedError(
                "only multiplication by scalars is supported for now"
            )
        new_tt = self.tt * other
        return self.__class__(
            new_tt, self.thresholds, self.categorical_features
        )

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            raise NotImplementedError(
                "only multiplication by scalars is supported for now"
            )
        new_tt = self.tt / other
        return self.__class__(
            new_tt, self.thresholds, self.categorical_features
        )

    def __add__(self, other):
        # Expand thresholds of both TTML so that they match
        self.expand_thresholds(other.thresholds)
        other.expand_thresholds(self.thresholds)

        # Add the underlying TT's
        new_tt = self.tt + other.tt
        new_tt.orthogonalize(force_rank=False)

        return self.__class__(
            new_tt, self.thresholds, self.categorical_features
        )

    def _new_tresh_median(self, X, mu, i):
        """Calculate median of a bin and number of points on either side of
        median."""
        right_bndry = self.thresholds[mu][i]
        if i == 0:
            left_bndry = -np.inf
        else:
            left_bndry = self.thresholds[mu][i - 1]
        X_filtered = X[:, mu][
            (left_bndry < X[:, mu]) * (X[:, mu] <= right_bndry)
        ]
        median = np.median(X_filtered)
        # Calculate minimum number of points in either new bin
        left_points = np.sum(X_filtered <= median)
        right_points = np.sum(X_filtered > median)
        min_split_points = min(left_points, right_points)

        return median, min_split_points

    def _expand_tresholds_from_residuals(
        self, X, y, n=5, min_split=5, task="regression"
    ):
        """Add thresholds inplace according to worst performance.

        Find the n worst-performing threshold in terms of contribution to the
        squared sum error or cross entropy, and split those thresholds in two
        according to median.

        Parameters
        ----------
        X : array<float, float>
        y : array<float>
        n : int
        min_split : int or float (default: 5)
            If an integer, only split a threshold if both new bins will contain
            at least `min_split` samples. If a float between 0 and 1, split
            based on fraction of all samples instead.
        """
        if min_split <= 0:
            min_split = 0
        elif 0 < min_split < 1:
            min_split = int(min_split * len(y))
        else:
            min_split = int(min_split)

        idx = self.observed_indices(X)
        predictions = self.predict(X)
        if task == "classification":
            predictions = ar.do("sigmoid", predictions)
        # residuals = (self.predict(X) - y) ** 2
        errors = []
        mu_i = []
        for mu in range(self.tt.order):
            # that way adding tasks just requires changing something in one place.
            if mu in self.categorical_features:
                continue
            # compute mean error
            for i in range(self.tt.dims[mu]):
                predictions_masked = predictions[idx[:, mu] == i]
                y_masked = y[idx[:, mu] == i]
                if task == "regression":
                    residuals = (predictions_masked - y_masked) ** 2
                    errors.append(ar.do("sum", residuals))
                if task == "classification":
                    p = predictions_masked
                    cross_entropy = -y_masked * ar.do("log", p) - (
                        1 - y_masked
                    ) * ar.do("log", 1 - p)
                    errors.append(ar.do("sum", cross_entropy))
                mu_i.append((mu, i))

        worst_thresh = np.argsort(errors)
        split_points = np.array(mu_i)[worst_thresh]
        new_thresholds = [[] for _ in range(self.tt.order)]
        for mu, i in split_points:
            new_tresh, min_split_points = self._new_tresh_median(X, mu, i)
            if min_split_points > min_split:
                new_thresholds[mu].append(new_tresh)
            if len(new_thresholds) >= n:
                break
        self.expand_thresholds(new_thresholds)

    @classmethod
    def fit(
        cls,
        X,
        y,
        estimator,
        X_val=None,
        y_val=None,
        task="regression",
        num_thresholds=50,
        tt_cross_its=5,
        max_rank=10,
        opt_steps=100,
        opt_tol=1e-5,
        tt_cross_method="dmrg",
        estimator_output="logit",
        verbose=False,
        _thresholds=None,
        _ttls_kwargs=None,
        _predict_fun=None,
    ):
        """
        Fit a TTML, using `estimator` for initialization.

        The tensor train is initialized using the DMRG TT-cross algorithm from
        the function values of the estimator. It is then further optimized
        to lower training loss using Riemannian conjugate gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training X data values
        y : np.ndarray
            Training truth labels. Should be 0/1 for classification.
        estimator :
            An `sklearn`-like estimator for fitting. If the estimator is not yet
            fitted, this calls the estimator's .fit function. For classification
            tasks, take note of the `estimator_output` keyword. If your
            estimator does not have an appropriate .predict method, then pass
            _predict_fun as a last resort. For classification we assume this
            outputs logits.
        X_val : np.ndarray (optional)
            Validation set to monitor during training
        y_val: np.ndarray (optional)
        task : str (default: "regression")
            Whether to use "regression" or "classification" as task. In the case
            of "classification" this estimator will output logits.
        num_thresholds : int (default: 50)
            The number of thresholds per feature to pick from the forest.
            Ignored if the `_thresholds` kwarg is specified.
        tt_cross_its : int (default: 5)
            Number of iterations for the TT-cross algorithm
        max_rank : int (default: 10)
            Maximum rank for the tensor train
        opt_steps : int (default: 100)
            Number of steps of Riemannian conjugate gradient descent to take
        opt_tol : float (default:  1e-5)
            After 3 steps of no relative improvement of at least `opt_tol`, the
            Riemannian conjugate gradient descent is stopped. If `X_val` is
            supplied then error is monitored on validation set, otherwise on
            training set.
        tt_cross_method : str (default: "dmrg")
            Whether to use "regular" or "dmrg" type TT-cross algorithm for
            initialization.
        estimator_output : str (default: "logit")
            For classification tasks, the output of the estimators' .predict
            function. Supported arguments are "logit" and "proba". If the
            estimator has a .predict_proba method, then this is ignored and
            .predict_proba is used instead.
        verbose : bool (default: False)
            If True, print convergence and debug information
        _thresholds : list[np.ndarray] (optional)
            Use this list of thresholds instead of inferring from the forest.
            Should be a list of arrays, one array per feature. The last element
            of each array is expected to be `np.inf`. TODO: update
        _ttls_kwargs : dict (optional)
            Keyword arguments to pass to the Riemannian conjugate gradient
            descent optimizer. See `TensorTrainLineSearch` for details.
        _predict_fun : method (optional)
        """

        # Obtain thresholds
        if _thresholds is None:
            thresholds = cls.thresholds_from_data(X, num_thresholds)
        else:
            thresholds = _thresholds

        if _predict_fun is not None:
            predict_fun = _predict_fun
        else:
            # Check if estimator is fitted, if not try to fit it.
            try:
                check_is_fitted(estimator)
            except NotFittedError:
                try:
                    estimator.fit(X, y)
                except AttributeError:
                    pass

            # determine the right prediction function for training
            if task == "regression":
                predict_fun = estimator.predict
            elif task == "classification":

                def pred_logit(X):
                    probs = proba_fun(X)
                    # Deal with sklearn's weird probability output
                    if len(probs.shape) > 1 and probs.shape[1] == 2:
                        probs = probs[:, 1]
                    probs = np.clip(probs, 1e-8, 1 - 1e-8)
                    return logit(probs)

                if hasattr(estimator, "predict_proba"):
                    proba_fun = estimator.predict_proba
                    predict_fun = pred_logit
                elif estimator_output == "proba":
                    proba_fun = estimator.predict
                    predict_fun = pred_logit
                else:
                    predict_fun = estimator.predict

            else:
                raise ValueError(f"Unknown task {task}")

        # Use tt-cross to initialize the TT
        tt = estimator_to_tt_cross(
            predict_fun,
            thresholds,
            verbose=verbose,
            max_rank=max_rank,
            max_its=tt_cross_its,
            method=tt_cross_method,
        )
        ttml = cls(tt, thresholds)

        # Report loss after tt-cross
        if verbose:
            # use validation set for reporting if possible, otherwise training
            if X_val is not None:
                X_report = X_val
                y_report = y_val
            else:
                X_report = X
                y_report = y
            if task == "regression":
                ttml_error = mean_squared_error(y_report, ttml.predict(X_report))
                tree_error = mean_squared_error(y_report, predict_fun(X_report))
            else:
                ttml_error = log_loss(y_report, expit(ttml.predict(X_report)))
                tree_error = log_loss(y_report, expit(predict_fun(X_report)))
            print(
                f"validation loss pre-optimization: {ttml_error:.4e}, "
                f"vs. baseline {tree_error:.4e}"
            )

        idx = ttml.observed_indices(X)
        if X_val is not None:
            idx_val = ttml.observed_indices(X_val)
        if _ttls_kwargs is None:
            _ttls_kwargs = {}
        if task == "classification":
            # Gradients for classification are tiny, this makes linesearch find
            # good step sizes.
            _ttls_kwargs["auto_scale"] = True
        ttls = TTLS(tt.copy(), y, idx, task=task, **_ttls_kwargs)
        history = {
            "train_loss": [],
            "val_loss": [],
            "step_size": [],
            "grad": [],
            "time": [],
        }
        start_time = perf_counter_ns()
        prev_loss = None
        num_steps_little_change = 0
        for i in range(opt_steps):
            try:
                train_loss, grad, step_size = ttls.step()
            except (LinAlgError, ValueError):
                # Linalg error points to convergence
                # On MKL this raises ValueError instead
                break
            if step_size is None:  # Line search didn't converge
                break
            history["train_loss"].append(train_loss)
            if X_val is not None:
                val_loss = ttls.loss(y=y_val, idx=idx_val)
                loss = val_loss
                history["val_loss"].append(val_loss)
                if verbose:
                    print(f"{i=}, {val_loss=:.4e}, {train_loss=:.4e}")
            else:
                loss = train_loss
                if verbose:
                    print(f"{i=}, {train_loss=:.4e}")
            grad = np.abs(grad) / len(y)
            history["grad"].append(grad)
            history["step_size"].append(step_size)
            time_delta = perf_counter_ns() - start_time
            history["time"].append(time_delta / 1e9)

            # Monitor relative improvement for early stopping
            if opt_tol is not None:
                if prev_loss is not None:
                    rel_error = (prev_loss - loss) / np.abs(prev_loss)
                else:
                    rel_error = np.inf
                prev_loss = loss
                if rel_error < opt_tol:
                    num_steps_little_change += 1
                else:
                    num_steps_little_change = 0
                if num_steps_little_change >= 3:
                    break
        ttml.tt = ttls.tt
        ttml._history = history
        return ttml

    def _fit_old(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        task="regression",
        sample_weight=None,
        n_steps_init=50,
        n_rounds=50,
        n_steps_round=20,
        n_thresh_round=10,
        n_round_rank_increase=5,
        min_improvement=1e-5,
        backend="numpy",
        optimizer_cls=TensorTrainLineSearch,
        opt_kwargs=None,
        verbose=False,
    ):
        """Fit the TTML to data using RGD, incrementing number of thresholds
        during training.

        Parameters
        ----------forest
        X : np.ndarray
        y : np.ndarray
        X_val : np.ndarray (optional)
            If supplied together with `y_val`, validation loss is monitored
            during training.
        y_val : np.ndarray or None (optional)
        task : str (default: 'regression')
            Whether to perform `regression` or `classification`
        sample_weight : np.ndarray
            Array of same shape as `y` giving weights to samples
        n_steps_init : int (default: 50)
            Number of steps to take before adding more thresholds
        n_rounds : int (default: 50)
            Number of times thresholds are added or rank is increased.
        n_steps_round : int (default: 20)
            Number of steps to take between adding thresholds
        n_thresh_round : int (default: 10)
            Number of thresholds to add each round
        n_round_rank_increase : int (default: 5)
            Increase rank every `n_round_rank_increase` rounds
        min_improvement : float (optional, default: 1e-5)
            If not `None`, stop optimization round if relative change in
            training loss is less than `min_improvement` for 3 consecutive steps
        backend : str (default: "numpy")
        optimizer : `TensorTrainOptimizer` (default: `TensorTrainLineSearch`)
            Which optimizer to use. By default use `TensorTrainLineSearch`,
            other supported option is `TensorTrainSGD`.
        opt_kwargs : dict or None (default: None)
            keyword arguments to pass to the optimizer.
        verbose : bool

        Returns
        -------
        history : dict
            Dictionary containing variables monitored during training:
            train_loss : training loss at each step
            val_loss : validation loss at each step (if `X_val` is not None)
            step_size : step size taken by optimizer
            grad : Riemannian gradient norm
            time : time in ms since start
        """
        backend = self.backend
        history = {
            "train_loss": [],
            "val_loss": [],
            "step_size": [],
            "grad": [],
            "time": [],
        }
        start_time = np.datetime64(datetime.now())

        def opt_round(n):
            prev_phi0 = None
            num_steps_little_change = 0

            idx = convert_backend(self.observed_indices(X), backend)
            if X_val is not None:
                idx_val = convert_backend(self.observed_indices(X_val), backend)

            # Make default step size equal to previous step size for faster start
            if len(history["step_size"]) > 0:
                last_step_size = history["step_size"][-1]
            else:
                last_step_size = 1.0

            opt = optimizer_cls(
                self.tt,
                y,
                idx,
                task=task,
                sample_weight=sample_weight,
                last_step_size=last_step_size,
                **opt_kwargs,
            )
            for _ in range(n):
                phi0, derphi0, step_size = opt.step()
                if step_size is None:
                    break
                history["train_loss"].append(phi0)
                if X_val is not None:
                    val_loss = opt.loss(y=y_val, idx=idx_val)
                    history["val_loss"].append(val_loss)
                grad = np.abs(derphi0) / len(y)
                history["grad"].append(grad)
                history["step_size"].append(step_size)
                time_delta = np.datetime64(datetime.now()) - start_time
                history["time"].append(time_delta / np.timedelta64(1, "ms"))

                # Monitor relative improvement for early stopping
                if min_improvement is not None:
                    if prev_phi0 is not None:
                        rel_error = (prev_phi0 - phi0) / np.abs(prev_phi0)
                    else:
                        rel_error = np.inf
                    prev_phi0 = phi0
                    if rel_error < min_improvement:
                        num_steps_little_change += 1
                    else:
                        num_steps_little_change = 0
                    if num_steps_little_change >= 3:
                        break

            self.tt = opt.tt

        try:
            opt_round(n_steps_init)
        except LinAlgError:
            print("LinalgError in initial round, reduce number of steps!")
        if verbose:
            if X_val is not None:
                val_string = f" Validation error: {history['val_loss'][-1]:.5f}"
            else:
                val_string = ""
            print(
                f"Initial round complete. Train error: {history['train_loss'][-1]:.5f}.{val_string}"
            )
        for i in range(n_rounds):
            self._expand_tresholds_from_residuals(
                X, y, n=n_thresh_round, task=task
            )
            if i > 0 and (i % n_round_rank_increase == 0):
                self.tt.increase_rank(1)
            try:
                opt_round(n_steps_round)
            except LinAlgError:
                if verbose:
                    print("Training aborted, probably convergence reached.")
                break
            if verbose:
                if X_val is not None:
                    val_string = (
                        f" Validation error: {history['val_loss'][-1]:.5f}"
                    )
                else:
                    val_string = ""
                print(
                    f"Round {i+1}/{n_rounds}. Train error: {history['train_loss'][-1]:.5f}.\
                    {val_string}"
                )
        return history


class TTMLEstimator(BaseEstimator):
    """Meta class to use TTML as sklearn estimator. Use derivative classes
    TTMLClassifier and TTMLRegressor instead.
    """

    def __init__(
        self,
        estimator,
        task=None,
        max_rank=5,
        num_thresholds=50,
        tt_cross_its=5,
        opt_steps=100,
        opt_tol=1e-5,
        verbose=False,
        **kwargs,
    ):
        self.estimator = estimator
        self.task = task
        self.max_rank = max_rank
        self.is_fitted = False
        self.verbose = verbose
        self.num_thresholds = num_thresholds
        self.tt_cross_its = tt_cross_its
        self.opt_steps = opt_steps
        self.opt_tol = opt_tol
        self.kwargs = kwargs

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.ttml_ = TTML.fit(
            X,
            y,
            self.estimator,
            X_val=X_val,
            y_val=y_val,
            task=self.task,
            num_thresholds=self.num_thresholds,
            tt_cross_its=self.tt_cross_its,
            max_rank=self.max_rank,
            opt_steps=self.opt_steps,
            opt_tol=self.opt_tol,
            verbose=self.verbose,
            **self.kwargs,
        )
        self.history_ = self.ttml_._history
        self.is_fitted = True
        return self


class TTMLRegressor(TTMLEstimator, RegressorMixin):
    __doc__ = (
        """Wrapper to turn TTML into an sklearn classifier."""
        + _fit_parameters_string
    )

    def __init__(self, estimator, **kwargs):
        super(TTMLRegressor, self).__init__(
            estimator, task="regression", **kwargs
        )

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = self.ttml_.predict(X)
        return prediction


class TTMLClassifier(TTMLEstimator, ClassifierMixin):
    __doc__ = (
        """Wrapper to turn TTML into an sklearn classifier."""
        + _fit_parameters_string
    )

    def __init__(self, estimator, **kwargs):
        super(TTMLClassifier, self).__init__(
            estimator, task="classification", **kwargs
        )

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = self.ttml_.predict(X, task="classification")
        return prediction

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = self.ttml_.predict(X, task="regression")
        prediction = ar.do("sigmoid", prediction)
        return prediction

    def predict_logit(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = self.ttml_.predict(X, task="regression")
        return prediction
