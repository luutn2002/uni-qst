from typing import Callable, Tuple
import tensorflow as tf
import numpy as np

from uni_qst.ms_nn.ops import convert_to_real_ops
from qutip import coherent_dm

def train_step(loss: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
               model,
               measurement: tf.Tensor, 
               label: tf.Tensor, 
               state: tf.Tensor, 
               ops: tf.Tensor,
               optimizer: tf.keras.optimizers.Optimizer) -> Tuple[float, float, float]:
    """
    Train step function of MS-NN.

    - Args:
        - loss(`function` or `tf.keras.Loss`): loss function, should be uni_qst.loss.MSNN_MSE_Loss.
        - model(`tf.keras.Model`): MS-NN model.
        - measurement(`tf.Tensor`): Training measurement tensors.
        - label(`tf.Tensor`): Training label tensors.
        - state(`tf.Tensor`): Training state tensors, performing measurement should be able to colaspe into `measurement` variable.
        - ops(`tf.Tensor`): Measurement ops.
        - optimizer(`tf.keras.Optimizer`): Optimizer function.

    - Return:
        - total_loss, real_loss, imag_loss(`float` or `tf.Tensor`): Loss values, with real + imag = total, ignore the last 2 return value if you don't need it .
    """
    with tf.GradientTape() as gen_tape:
        
        generated_state, generated_measurement = model([ops, label, measurement], training=True)

        total_loss, real_loss, imag_loss = loss(generated_measurement, measurement, generated_state, state)
     

    gradients = gen_tape.gradient(
        total_loss, model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    return total_loss, real_loss, imag_loss

def create_tf_training_ops(batch_size,
                           hilbert_size = 32,
                           grid = 32):
    """
    Create measurement ops for model.

    - Args:
        - batch_size(`int`): Inference batch size.
        - hilbert_size(`int`): Hilbert size of the state.
        - grid(`int`): Grid size of the state.

    - Return:
        - ops_tf(`tf.Tensor`): Measurement op for MS-NN to be used in inference and training                          .
    """
    hilbert_size = 32
    grid = 32
    xvec = np.linspace(-5, 5, grid)
    yvec = np.linspace(-3, 3, grid)

    X, Y = np.meshgrid(xvec, yvec)
    betas = (X + 1j*Y).ravel()
    m_ops = [(1/np.pi)*coherent_dm(hilbert_size, beta) for beta in betas]

    ops_numpy = [op.full() for op in m_ops] # convert the QuTiP Qobj to numpy arrays
    ops_tf = tf.convert_to_tensor([ops_numpy])
    ops_tf = convert_to_real_ops(ops_tf)
    ops_tf = tf.tile(ops_tf, tf.constant([batch_size, 1, 1, 1], tf.int32))
    return ops_tf
