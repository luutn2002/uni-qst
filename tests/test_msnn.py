import pytest
from uni_qst.ms_nn.model import MSNN
from uni_qst.ms_nn.utils import *
from uni_qst.loss import MSNN_MSE_Loss
from uni_qst.dataset import DatasetGenerator
import tensorflow as tf

def test_MSNN(batch_size = 2):
        try:
                generator = DatasetGenerator()
                train_data, train_label, train_value, _, _, _ , _, _ = generator.generate(dataset_size=10, to_numpy = True, save_file = False)
                train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label, train_value))
                train_dataset = train_dataset.batch(batch_size)
                initial_learning_rate = 1e-4
                optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
                model = MSNN()
                ops = create_tf_training_ops(batch_size)
                for measurement, label, state in train_dataset:
                        train_step(MSNN_MSE_Loss, model, measurement, label, state, ops, optimizer)
        except Exception as e:
                pytest.fail(f"{e}")
                
                
