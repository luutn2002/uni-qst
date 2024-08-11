"""
Tensorflow ops for MS-NN. 
"""
from typing import Tuple
import tensorflow as tf


def batched_expect(ops : tf.Tensor, 
                   rhos: tf.Tensor)-> tf.Tensor:
    """
    Calculates expectation values for a batch of density matrices
    for a set of operators.
    - Args:
        - ops (`tf.Tensor`): a tensor of shape (batch_size, N, hilbert_size, hilbert_size) of N measurement operators.
        - rhos (`tf.Tensor`): a tensor (batch_size, hilbert_size, hilbert_size).
    - Returns:
        - expectations (:class:`tf.Tensor`): A tensor shaped as (batch_size, N) representing expectation values for the N operators for all the density matrices  (batch_size).
    """
    products = tf.einsum("bnij, bjk->bnik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations

def clean_cholesky(img: tf.Tensor)-> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create 2 positive semidefinite Hermitian matrix
    - Args:
        - img (`tf.Tensor`): a tensor of shape (batch_size, hilbert_size, hilbert_size, 2) representing random outputs from a neural network. The last dimension is for separating the real and imaginary part.
    - Returns:
        - (T1, T2) (`tuple(tf.Tensor, tf.Tensor)`): a 3D tensor (N, hilbert_size, hilbert_size) representing N matrices used for Cholesky decomposition.
    """
    real = img[:, :, :, 0]
    imag = img[:, :, :, 1]

    diag_all = tf.linalg.diag_part(imag, k=0, padding_value=0)
    diags = tf.linalg.diag(diag_all)

    imag = imag - diags
    imag1 = tf.linalg.band_part(imag, -1, 0)
    real1 = tf.linalg.band_part(real, -1, 0)
    
    imag2 = tf.linalg.band_part(imag, 0, -1)
    real2 = tf.linalg.band_part(real, 0, -1)
    
    T1, T2 = tf.complex(real1, imag1) , tf.complex(real2, imag2)
    return T1, T2


def density_matrix_from_T(T1: tf.Tensor, T2: tf.Tensor)-> tf.Tensor:
    """
    Gets density matrices from upper and lower conjugated matrices and normalizes them.
    - Args:
        - T1 (`tf.Tensor`): A tensor (N, hilbert_size, hilbert_size) representing N valid upper conjugated matrices.
        - T2 (`tf.Tensor`): A tensor (N, hilbert_size, hilbert_size) representing N valid lower conjugated matrices.
    - Returns:
        - rho (`tf.Tensor`): A tensor of shape (N, hilbert_size, hilbert_size) representing N density matrices.
    """
    T1_dagger = tf.transpose(T1, perm=[0, 2, 1], conjugate=True)
    T2_dagger = tf.transpose(T2, perm=[0, 2, 1], conjugate=True)
    proper_dm = tf.math.subtract(tf.matmul(T1_dagger, T1), tf.matmul(T2, T2_dagger))
    all_traces = tf.linalg.trace(proper_dm)
    all_traces = tf.reshape(1 / all_traces, (-1, 1))
    rho = tf.einsum("bij,bk->bij", proper_dm, all_traces)
    return rho

def convert_to_complex_ops(ops: tf.Tensor)-> tf.Tensor:
    """
    Converts a batch of TensorFlow operators to something that a neural network can take as input.
    - Args:
        - ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N measurement operators.
    - Returns:
        - tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N measurement operators converted into real matrices.
    """
    shape = ops.shape
    num_points = shape[-1]
    tf_ops = tf.complex(
        ops[..., : int(num_points / 2)], ops[..., int(num_points / 2) :]
    )
    tf_ops = tf.transpose(tf_ops, perm=[0, 3, 1, 2])
    return tf_ops


def tf_fidelity(A: tf.Tensor, B: tf.Tensor)-> tf.Tensor:
    """
    Calculates the fidelity between tensors A and B.
    - Args:
        - A (`tf.Tensor`): List of tensors (hilbert_size, hilbert_size).
        - B (`tf.Tensor`): List of tensors (hilbert_size, hilbert_size).
    - Returns:
        - fidel (`float`): Fidelity between A and B.
    """
    sqrtmA = tf.matrix_square_root(A)
    temp = tf.matmul(sqrtmA, B)
    temp2 = tf.matmul(temp, sqrtmA)
    fidel = tf.linalg.trace(tf.linalg.sqrtm(temp2)) ** 2
    fidel = tf.math.real(fidel)
    fidel = tf.where(fidel < 0., 0., fidel)
    fidel = tf.where(fidel > 1., 0., fidel)
    fidel = tf.reduce_mean(fidel)
    return fidel

def convert_to_real_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network can take as input.
    - Args:
        - ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N measurement operators.
    - Returns:
        - tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N measurement operators converted into real matrices.
    """
    tf_ops = tf.transpose(ops, perm=[0, 2, 3, 1])
    tf_ops = tf.concat([tf.math.real(tf_ops), tf.math.imag(tf_ops)], axis=-1)
    return tf_ops
