"""
Basics
~~~~~~

Tensor trains are a versatile tensor decomposition. They consist of a list of
order-3 tensors known as `cores`. A tensor train encoding an order `d` dense
tensor, has `d` cores. The second dimension of these cores coincides with the
dimensions of the dense tensor. The first and third dimensions of the cores are
known as the `tensor train rank`.

For example, below we create a random tensor train encoding a shape ``(10, 12,
8)`` tensor, with tensor train ranks ``(3, 4)``

>>> from ttml.tensor_train import TensorTrain
...
... dims = (10, 12, 8)
... tt_rank = (3, 4)
... tt = TensorTrain.random(dims, tt_rank)
... tt
<TensorTrain of order 3 with outer dimensions (10, 12, 8), TT-rank (3, 4),
and orthogonalized at mode 2>

We can access the cores of the tensor train through simple array indexing or
looping. Let's print the shapes of the three cores in this tensor train

>>> for core in tt
...     print(core.shape)
(1, 10, 3)
(3, 12, 4)
(4, 8, 1)

Note that the first dimension of the first core and the last dimension of the
last core is always ``1``.

Orthogonalization
~~~~~~~~~~~~~~~~~

Note in the representation string above, it is mentioned that the tensor train
is ``orthogonalized at mode 2``. This means that the `left-matricization` of the
cores to the left of mode 2 (i.e. the first and second cores) are orthogonal.
This means that the following contraction is the identity matrix:

>>> import numpy as np
...
... np.einsum("abi,abj->ij", tt[0], tt[0])
array([[ 1.00000000e+00, -8.84708973e-17,  3.46944695e-18],
       [-8.84708973e-17,  1.00000000e+00, -6.93889390e-18],
       [ 3.46944695e-18, -6.93889390e-18,  1.00000000e+00]])

The same contraction is also an identity matrix for ``tt[1]``. Orthogonalization
is extremely important for numerical stability, as well as for working with
tangent vectors (more on that later). We can specify on which mode we want the
tensor train to be orthogonalized at initialization, but we can also change the
orthogonalization mode by using the :meth:`TensorTrain.orthogonalize` method.

For example, we can orthogonalize the tensor train on mode 1 below. The mode
argument can be any integer, or either of the strings ``'l'`` and ``'r'``. Here
``'l'`` corresponds to a `left orthogonalization`, which is an orthogonalization
with respect to `the last mode`. Conversely ``'r'`` corresponds to a `right
orthogonalization`, that is, with respect to `the first mode`.

>>> tt.orthogonalize(mode=1)
...
... A = np.einsum("abi,abj->ij", tt[0], tt[0])
... print(np.allclose(A, np.eye(len(A))))
...
... B = np.einsum("iab,jab->ij", tt[2], tt[2])
... print(np.allclose(B, np.eye(len(B))))
True
True

One consequence of orthogonalization is that computing the norm of the the
tensor train can be done very efficiently; the (Frobenius) norm of a tensor
trained orthogonalized at mode ``mu`` is simply the (Frobenius) norm of core
``mu``.

>>> np.isclose(tt.norm(), np.linalg.norm(tt[1]))
True

Tensor train arithmetic
~~~~~~~~~~~~~~~~~~~~~~~

We can also perform many operations involving two or more tensor trains. Let's
first create two new tensor trains with the same outer dimensions, but different
tt-ranks. We can then for example contract the tensor trains using the
:meth:`TensorTrain.dot` method, or alternatively the ``@`` operator.

>>> dims = (4, 6, 6, 5)
... tt1 = TensorTrain.random(dims, (3, 4, 3))
... tt2 = TensorTrain.random(dims, (2, 2, 2))
... tt1 @ tt2
0.016860730327956833

We can verify that this is indeed the Frobenius inner product of these two
tensors by comparing the result to contracting the two associated dense tensors.
We can turn a tensor train into a dense tensor using the
:meth:`TensorTrain.dense` method.

>>> np.einsum("ijkl,ijkl->", tt1.dense(), tt2.dense())
0.016860730327956833

We can also add/subtract tensor trains or multiply them by scalars.

>>> tt3 = tt1 + 0.1 * tt2
... tt3
<TensorTrain of order 4 with outer dimensions (4, 6, 6, 5), TT-rank (4, 6, 5),
and orthogonalized at mode 3>

Truncation
~~~~~~~~~~

Note that when we add tensor trains, the outer dimensions stay the same but in
principle the tt-rank increases, becoming at most the sum of the tt-ranks.
In many cases the rank of the sum is not the sum of the ranks. For example in
the case above, the first rank of ``tt3`` is 4 and not 5=2+3, since the first
rank is always bounded by the first dimension (which is 4 in this case).

If we now add another copy of ``tt2`` to ``tt3`` we would expect the rank to
stay the same, yet this doesn't always happen due to numerical errors. Note
that the middle tt-rank below is 8, even though ``tt4`` can be expressed
by a rank ``(4, 6, 5)`` tensor train.

>>> tt4 = tt3 + tt2
... tt4
<TensorTrain of order 4 with outer dimensions (4, 6, 6, 5), TT-rank (4, 8, 5),
and orthogonalized at mode 3>

We can truncate the rank of a tensor train by using the
:meth:`TensorTrain.round` method. This uses HOSVD to truncate the tensor train,
and it has two methods for rounding; it can round to a pre-specified tt-rank,
or it can truncate based on singular values. We do the latter below by
specifying the ``eps=1e-16`` keyword, meaning we can round each core in a HOSVD
sweep with relative error up to ``1e-16``. 

>>> tt5 = tt4.round(eps=1e-16, inplace=False)
... print(tt5)
... (tt4 - tt5).norm()
<TensorTrain of order 4 with outer dimensions (4, 6, 6, 5), TT-rank (4, 6, 5),
and orthogonalized at mode 3>
3.1508721275986887e-15

Note that the rank has decreased to the correct value, while only gathering an
error on the order of machine epsilon. This is because the last two singular
values of the second unfolding are very small, we can see them using
:meth:`TensorTrain.sing_vals`:

>>> tt4.sing_vals()
[array([1.92032396, 1.1634468 , 0.62421987, 0.4174046 ]),
 array([1.77671233e+00, 9.81237260e-01, 7.97148647e-01, 6.93386512e-01,
        5.05833036e-01, 3.36895334e-01, 2.17444526e-31, 1.32445319e-31]),
 array([1.95129345, 0.99099654, 0.80976405, 0.38830473, 0.09492614])]

We can also round to even lower ranks, at the cost of a higher rounding error.

>>> tt6 = tt4.round(max_rank=4, inplace=False)
... (tt4 - tt6).norm()
0.6129466966082833

Here ``max_rank=4`` means all tt-ranks should be at most 4, but we could also
supply a tuple of ints here, e.g. ``tt4.round(max_rank = (4, 5, 5))``.

Accessing entries
~~~~~~~~~~~~~~~~~

To access any specific entries of the tensor train we can use the
:meth:`TensorTrain.gather` method. We need to supply it a list of entries
we want to access, encoded as an integer array. For example to access entry
(0,0,0,0) and (0,1,0,0) we do the following:

>>> tt4.gather(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]))
array([ 0.02875322, -0.02476423])

Tangent vectors
~~~~~~~~~~~~~~~
For Riemannian optimization of tensor trains we need to work with tangent
vectors on the manifold of tensor trains of specified rank. Tangent vectors
are always associated to a particular tensor train (remember: a tensor train
is a point on the tensor-train manifold). For efficient manipulation of
tangent vectors, we need to have both the left- and right-orthogonalized cores
of a tensor train. Since tensor trains are left-orthogonalized by default, we
just need to compute the right-orthogonal cores. 

>>> tt = TensorTrain.random((3, 4, 4, 3), (2, 3, 2))
... right_cores = tt.orthogonalize(mode="r", inplace=False)
... tv = TensorTrainTangentVector.random(tt, right_cores)
... tv
<TensorTrainTangentVector of order 4, outer dimensions (3, 4, 4, 3),
and TT-rank (2, 3, 2)>

The arguably most important thing we can do with tangent vectors is that using
a `'retract'` we can 'move' in the direction of the tangent vector. We can do
this using the :meth:`TensorTrain.apply_grad` method. Below we apply the retract
of ``tv`` to `tt`, after first multiplying it by ``1e-6`` using the ``alpha``
keyword argument.

>>> tt2 = tt.apply_grad(tv, alpha=1e-6)
... (tt-tt2).norm()
8.200339164343568e-07

We see that this changes `tt` on the order of the 'step size' ``alpha`` and the
norm of ``tt2``.

Transporting a tangent vector to a new point is equivalent to projecting the
tangent vector to the tangent space of the new point. This can be done using
:meth:`TensorTrain.grad_proj`:

>>> tv2 = tt2.grad_proj(tv)
<TensorTrainTangentVector of order 4, outer dimensions (3, 4, 4, 3),
and TT-rank (2, 3, 2)>

We can also perform arithmetic with tangent vectors, like multiplying them by
scalars, adding them, or computing inner products (using
:meth:`TensorTrainTangentVector.inner` or the ``@`` operator).
Note that this only makes mathematical sense if the tangent vectors are
associated to the same tensor train.

>>> tv1 = TensorTrainTangentVector.random(tt, right_cores)
... tv2 = TensorTrainTangentVector.random(tt, right_cores)
... print((tv1 + tv2).norm())
... tv3 = tv1 - 0.1 * tv2
... print(tv1 @ tv3)
1.278782626647999
0.6505614686324931

The way we usually create tangent vectors is as the gradient of some
optimization problem. For example we could try to solve a tensor completion
problem. We want to approximate an unknown dense tensor using a tensor train
based on knowing the values of particular entries of the dense tensor. We can
solve this by starting with a random tensor train and applying Riemannian
optimization. At each point the `Euclidean gradient` is just linear residual
error between the entries in the tensor train and the true value. We can convert
this Euclidean gradient into a tangent vector using
:meth:`TensorTrain.rgrad_sparse`. Below we illustrate one step of gradient
descent for the tensor completion problem.

>>> tt = TensorTrain.random((10, 10, 10, 10, 10), (2, 5, 5, 2))
...
... # Generate 100 random values and 100 random indices of `tt`
... N = 100
... y = np.random.normal(N)
... idx = [np.random.choice(r, size=N) for r in tt.tt_rank]
... idx = np.stack(idx)
...
... # Compute the initial error and the gradient
... prediction = tt.gather(idx)
... residual = prediction - y
... print(np.linalg.norm(residual))
... grad = tt.rgrad_sparse(-residual, idx)
...
... # Take a step in gradient direction and compute new error
... tt2 = tt.apply_grad(grad, alpha=10)
... prediction = tt2.gather(idx)
... residual = y - prediction
... print(np.linalg.norm(residual))
initial error: 200.76000727481116
error after step: 191.21863957535254


Alternative backends
~~~~~~~~~~~~~~~~~~~~

So far all the objects we have use were encoded as numpy arrays, but other
backends are supported as well. For example to use ``tensorflow`` as a backend
for a tensor train, we just need to supply the ``backend`` keyword:

>>> tt_tf = TensorTrain.random((4, 4, 4), (2, 2), backend="tensorflow")
... print(tt_tf[0])

Here in principle many backends are supported, such as ``pytorch``, ``dask``,
``jax`` or ``cupy``. Support for other backends is handled by
`autoray <https://github.com/jcmgray/autoray>`_. However, not all functionality
has been thoroughly tested for most backends. Moreover, all the functions used
here are not `compiled`, so usually things end up being fastest for numpy. This
may change in the future. In particular :class:`TTML` has very limited
support for backends other than numpy, since the ``scikit-learn`` estimators
used for initialization only support numpy anyway.
"""
