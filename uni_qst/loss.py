"""
Loss functions used in experiments.
"""
import tensorflow as tf
from torch import nn

def MSNN_MSE_Loss(generated_measurement: tf.Tensor, 
                  measurement: tf.Tensor, 
                  generated_state: tf.Tensor, 
                  state: tf.Tensor, 
                  lam=100.):
    """
    Computes the weighted MSE loss sum of MS-NN.
    - Args:
        - generated_measurement (tf.Tensor): Husimi-Q measurements of MS-NN output states.
        - measurement (tf.Tensor): Husimi-Q measurements of dataset target states.
        - generated_state (tf.Tensor): MS-NN output states.
        - state (tf.Tensor): Dataset target states.
        - lam (float, optional): The weight of the loss. Default is 100.
    - Returns:
        - total_gen_loss, real_loss, imag_loss (float, float, float): Loss terms, include total loss, real part and imaginary part loss of the density matrix.
    """

    # mean absolute error
    mea_loss = tf.reduce_mean(tf.abs(measurement - generated_measurement))
    real_loss = tf.reduce_mean(tf.abs(tf.math.real(state) - tf.math.real(generated_state)))
    imag_loss = tf.reduce_mean(tf.abs(tf.math.imag(state) - tf.math.imag(generated_state)))
    
    total_gen_loss = mea_loss + lam*(real_loss + imag_loss)

    return total_gen_loss, real_loss, imag_loss

class LinearCombinationLoss(nn.Module):
    def __init__(self):
        super(LinearCombinationLoss, self).__init__()

    def forward(self, label_output, label_target, value_output, value_target):
        return nn.CrossEntropyLoss()(label_output, label_target) + nn.L1Loss()(value_output.real, value_target.real) + \
    nn.L1Loss()(value_output.imag, value_target.imag)