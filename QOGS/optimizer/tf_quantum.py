#%%
# Code here is various things taken from Henry Liu and Vlad Sivak as well as some of my own additions
# Keep up to date with their versions, and later just import from their
# soon to come tf-quantum library.
from distutils.version import LooseVersion

import tensorflow as tf
import qutip as qt

if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np

    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)

#%%
# todo: handle case when passed an already tf object.
def qt2tf(qt_object, dtype=tf.complex64):
    if tf.is_tensor(qt_object) or qt_object is None:
        return qt_object
    return tf.constant(qt_object.full(), dtype=dtype)


def tf2qt(tf_object, matrix=False):
    if not tf.is_tensor(tf_object):
        return tf_object

    return qt.Qobj(
        tf_object.numpy(),
        dims=[
            [2, int(tf_object.numpy().shape[0] / 2)],
            [1, 1] if not matrix else [2, int(tf_object.numpy().shape[0] / 2)],
        ],
    )


def matrix_flatten(tensor):
    """Takes a tensor of arbitrary shape and returns a "flattened" vector of matrices.
    This is useful to get the correct broadcasting shape for batch operations.
    Args:
        tensor (Tensor([x, y, ...])): A tensor with arbitrary shape
    Returns:
        Tensor([numx * numy * ..., 1, 1]): A "flattened" vector of matrices
    """
    tensor = tf.reshape(tensor, [-1])
    tensor = tf.reshape(tensor, shape=[tensor.shape[0], 1, 1])
    return tensor


def identity(N, dtype=tf.complex64):
    """Returns an identity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN identity operator
    """
    return tf.eye(N, dtype=dtype)


def destroy(N, dtype=tf.complex64):
    """Returns a destruction (lowering) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float64)), k=1)
    return tf.cast(a, dtype=dtype)


def create(N, dtype=tf.complex64):
    """Returns a creation (raising) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    # Preserve max precision in intermediate calculations until final cast
    return tf.cast(tf.linalg.adjoint(destroy(N, dtype=tf.complex128)), dtype=dtype)


def num(N, dtype=tf.complex64):
    """Returns the number operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=dtype)


def position(N, dtype=tf.complex64):
    """Returns the position operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast((a_dag + a) / sqrt2, dtype=dtype)


def momentum(N, dtype=tf.complex64):
    """Returns the momentum operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], c64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=dtype)


def kron(x, y):
    """Computes the Kronecker product of two matrices.

  Args:
    x: A matrix (or batch thereof) of size m x n.
    y: A matrix (or batch thereof) of size p x q.

  Returns:
    z: Kronecker product of matrices x and y of size mp x nq
  """
    with tf.name_scope("kron"):
        x = tf.convert_to_tensor(x, dtype_hint=tf.float32)
        y = tf.convert_to_tensor(y, dtype_hint=x.dtype)

        def _maybe_expand(x):
            xs = tf.pad(
                tf.shape(x),
                paddings=[[tf.maximum(2 - tf.rank(x), 0), 0]],
                constant_values=1,
            )
            x = tf.reshape(x, xs)
            _, mx, nx = tf.split(xs, num_or_size_splits=[-1, 1, 1])
            return x, mx, nx

        x, mx, nx = _maybe_expand(x)
        y, my, ny = _maybe_expand(y)
        x = x[..., :, tf.newaxis, :, tf.newaxis]
        y = y[..., tf.newaxis, :, tf.newaxis, :]
        z = x * y
        bz = tf.shape(z)[:-4]
        z = tf.reshape(z, tf.concat([bz, mx * my, nx * ny], axis=0))
        return z


def tf_kron(a, b):
    a_shape = a.shape  # [a.shape[0], a.shape[1]]
    b_shape = b.shape  # [b.shape[0], b.shape[1]]
    if len(a_shape) == 2:
        return tf.reshape(
            tf.reshape(a, [a_shape[0], 1, a_shape[1], 1])
            * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
            [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]],
        )
    elif len(a_shape) == 3:
        return tf.reshape(
            tf.reshape(a, [a_shape[0], a_shape[1], 1, a_shape[2], 1])
            * tf.reshape(b, [b_shape[0], 1, b_shape[1], 1, b_shape[1]]),
            [a_shape[0], a_shape[1] * b_shape[1], a_shape[2] * b_shape[2]],
        )
    elif len(a_shape) == 4:
        return tf.reshape(
            tf.reshape(a, [a_shape[0], a_shape[1], a_shape[2], 1, a_shape[3], 1])
            * tf.reshape(b, [b_shape[0], b_shape[1], 1, b_shape[2], 1, b_shape[3]]),
            [a_shape[0], a_shape[1], a_shape[2] * b_shape[2], a_shape[3] * b_shape[3]],
        )


def two_mode_op(op, mode=0):
    """takes single mode op and returns op in two mode Hilbert space
    """
    N = op.shape[-1]
    if len(op.shape) == 2:
        I = tf.eye(N, N, batch_shape=None, dtype=op.dtype)
    elif len(op.shape) == 3:
        I = tf.eye(N, N, batch_shape=op.shape[0], dtype=op.dtype)
    elif len(op.shape) == 4:
        I = tf.eye(N, N, batch_shape=(op.shape[0], op.shape[1]), dtype=op.dtype)
    if mode == 0:
        op2 = tf_kron(op, I)
    else:
        op2 = tf_kron(I, op)
    # return tf.constant(op2, dtype=op.dtype)
    return op2


# %%
