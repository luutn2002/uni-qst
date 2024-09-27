# uni_qst: The official repository of "Universal Quantum Tomography with Deep Neural Networks"

[Paper](https://arxiv.org/abs/2407.01734)

## Installation

Installing this git repo as a package is recommended, you can simply install in Conda environment by:

```bash
$ conda install --yes -c pytorch pytorch=2.1.0 torchvision cudatoolkit=11.8
$ pip install git+https://github.com/luutn2002/uni-qst.git
```
## Generating the dataset used in paper

To create the dataset used from our experiment, simply call:

```python
from uni_qst.dataset import DatasetGenerator

generator = DatasetGenerator()
generator.generate(dataset_size=10000, #Total number of generated states.
                    train_split=0.8, #Training size split, 80% by default.
                    to_numpy = True, #Use numpy format to generate, if set to False then default to Torch .pt file.
                    save_file = True, #Save to train_dir and test_dir parameter directory, if set to False then return the data instead of saving. 
                    apply_noise = False, #Apply noises to dataset.
                    train_dir = './train', #Saved training data dir.
                    test_dir='./test') #Saved testing data dir.
```

To load generated data to tensorflow pipeline:

```python
import numpy as np
import tensorflow as tf

train_x = np.load('train/data.npy')
train_y = np.load('train/label.npy')
train_z = np.load('train/values.npy')

test_x = np.load('test/data.npy')
test_y = np.load('test/label.npy')
test_z = np.load('test/values.npy')

tf.keras.backend.set_floatx('float64')

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_z))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_z))
```

## Using RFB-Net and MS-NN in your experiments

To use MS-NN for training, you can simply call:

```python
from uni_qst.ms_nn.model import MSNN
from uni_qst.ms_nn.ops import tf_fidelity
from uni_qst.ms_nn.utils import *
from uni_qst.loss import MSNN_MSE_Loss
import tensorflow as tf

batch_size = 2

train_x = np.load('train/data.npy')
train_y = np.load('train/label.npy')
train_z = np.load('train/values.npy')

tf.keras.backend.set_floatx('float64')

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_z))
train_dataset = train_dataset.batch(batch_size)

initial_learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
model = MSNN()
ops = create_tf_training_ops(batch_size)

for measurement, label, state in train_dataset:
    train_step(MSNN_MSE_Loss, model, measurement, label, state, ops, optimizer)
    dm, _ = model([ops, label, measurement])
    fidelity = tf_fidelity(dm, values) #Recieved quantum fidelity here.
    ... #Do some other postprocessing.
```

To use RFB-Net for training, you can simply call:

```python
import torch
from uni_qst.rfb_net.model import RFBNet
from uni_qst.loss import LinearCombinationLoss
from uni_qst.dataset import CustomTensorDataset, DatasetGenerator
from torch.utils.data import DataLoader
from uni_qst.rfb_net.utils import train_step

train_data = torch.load('train/data.pt')
train_label = torch.load('train/label.pt')
train_value = torch.load('train/values.pt')
train_ft = torch.load('train/ft.pt')

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100

with device:

        train_dataset = CustomTensorDataset((train_data, train_label, train_value, train_ft))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        loss = LinearCombinationLoss().to(device)
        model = RFBNet()
        model.double().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for t in range(epochs):
            train_step(loss, model, train_dataloader, optimizer, device)
            ... #Do some postprocessing.
```

## References

This work is based largely from QST-NN and QST-CGAN. Original repo to QST-NN and QST-CGAN can be found [here](https://github.com/quantshah/qst-nn) and [here](https://github.com/quantshah/qst-cgan)

## Citation

If you find our repo useful, please cite as below:

```bibtex
@article{luu2024universal,
  title={Universal Quantum Tomography With Deep Neural Networks},
  author={Luu, Nhan T and Thang, Truong Cong and Duong T. Luu},
  journal={arXiv preprint arXiv:2407.01734},
  year={2024}
}
```
