"""
MS-NN layers and model function.
"""
from uni_qst.ms_nn.ops import (
    clean_cholesky,
    density_matrix_from_T,
    batched_expect,
    convert_to_complex_ops,
)
import tensorflow as tf

class DensityMatrix(tf.keras.layers.Layer):
    """
    Density matrix layer that cleans the input matrix into a Cholesky matrix and then constructs the density matrix for the state.
    """

    def __init__(self):
        super(DensityMatrix, self).__init__()

    def call(self, inputs: tf.Tensor)-> tf.Tensor:
        """
        The call function which applies the Cholesky decomposition.
        - Args:
            - inputs (`tf.Tensor`): a 4D real valued tensor (batch_size, hilbert_size, hilbert_size, 2) representing batch_size random outputs from a neural netowrk. The last dimension is for separating the real and imaginary part.
        - Returns:
            - dm (`tf.Tensor`): A 3D complex valued tensor (batch_size, hilbert_size, hilbert_size) representing valid density matrices from a Cholesky decomposition of the cleaned input.
        """
        T1, T2 = clean_cholesky(inputs)
        return density_matrix_from_T(T1, T2)


class Expectation(tf.keras.layers.Layer):
    """
    Expectation layer that calculates expectation values for a set of operators on a batch of rhos. You can specify different sets of operators for each density matrix in the batch.
    """
    def __init__(self):
        super(Expectation, self).__init__()

    def call(self, 
             ops: tf.Tensor, 
             rhos: tf.Tensor)-> tf.Tensor:
        """
        Expectation function call.
        - Args:
            - ops (`tf.Tensor`): a 4D complex tensor (batch_size, N, hilbert_size, hilbert_size) of N measurement operators.
            - rhos (`tf.Tensor`): a 4D complex tensor (batch_size, hilbert_size, hilbert_size).
        - Returns:
            - expectations (`tf.Tensor`): A 2D tensor (batch_size, N) giving expectation values for the N grid of operators for all the density matrices (batch_size).
        """
        return batched_expect(ops, rhos)

def MSNN(hilbert_size:int = 32, 
         num_class:int = 7)-> tf.keras.Model:
    """
    Build and return a MS-NN model.
        - Args:
            - hilbert_size (`int`): Hilbert dimension of input measurements, default to 32.
            - num_class (`int`): number of quantum state classes, default to 7.
        - Returns:
            - model (`tf.keras.Model`): A tf.keras.Model os MS-NN.
    """
    initializer = tf.random_normal_initializer(0.0, 0.001)
    
    ops = tf.keras.layers.Input(
        shape=[hilbert_size, hilbert_size, hilbert_size**2*2], name="operators")
    
    label = tf.keras.layers.Input(
        shape=1, name="label")
    
    measurement = tf.keras.Input(
        shape=[hilbert_size, hilbert_size], name="measurement")
    
    flatten_measurement = tf.keras.layers.Reshape((1, 1024))(measurement)
    encoded_label = tf.one_hot(tf.cast(label, tf.int32), num_class)
    
    x = tf.keras.layers.Dense(
        hilbert_size**2,
        use_bias=False,
        kernel_initializer=initializer,
    )(encoded_label) + flatten_measurement
    x = tf.keras.layers.Conv2DTranspose(
        256, (4, 4), use_bias=False, strides=1, kernel_initializer=initializer
    )(tf.expand_dims(x, 2))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), use_bias=False, strides=1, kernel_initializer=initializer
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(
        64, (4, 4), use_bias=False, strides=2, kernel_initializer=initializer
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(
        2, (2, 2), use_bias=False, strides=2, kernel_initializer=initializer
    )(x)

    rho = DensityMatrix()(x)
    complex_ops = convert_to_complex_ops(ops)
    x = Expectation()(complex_ops, rho)
    x = tf.keras.layers.Reshape((32, 32))(x)
    
    return tf.keras.Model(inputs=[ops, label, measurement], outputs=[rho, x])