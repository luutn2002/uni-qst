from torch import nn
import torch
import scipy as scp

class BatchedFidelity(nn.Module):
    def __init__(self):
        super(BatchedFidelity, self).__init__()
        '''
        Fidelity score of 2 state.
        '''
    def sqrt_mat(self, A):
        assert len(A.shape) == 3, 'This function only take 3 dim input'
        for img in A:
            img = torch.tensor(scp.linalg.sqrtm(torch.squeeze(img).numpy()), dtype=torch.complex128)
        return A
        
    def forward(self, A, B):
        sqrtmA = self.sqrt_mat(A)
        temp = torch.matmul(torch.matmul(sqrtmA, B), sqrtmA)
        fidel = self.sqrt_mat(temp).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        fidel = fidel.real
        fidel = torch.sqrt(fidel)
        return fidel
    
def train_step(loss_fn: nn.Module,
               model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: nn.Module,
               device: torch.device) -> float:
    """
    Train step function of RFB-Net

    - Args:
        - loss_fn(`nn.Module`): loss function, should be a torch.nn.Moule.
        - model(`nn.Module`): RFB-Net model.
        - dataloader(`torch.utils.data.DataLoader`): Torch dataloader use for training.
        - optimizer(`tf.keras.Optimizer`): Optimizer function.
        - device(`torch.device`): Training device.

    - Return:
        - loss(`float`): Loss value.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (data, data_label, value, feature) in enumerate(dataloader):
        data, label, value, feature = data.to(device), data_label.to(device),\
                                            value.to(device), feature.to(device)
        pred_label , pred_feature = model(data)
        
        loss = loss_fn(pred_label, label, pred_feature, feature)
        
        # Backprop classifier
        model.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 16.)
        optimizer.step()

        if batch % 2 == 0:
            current = batch * len(data)
            print(f"Current batch: [{current:>5d}/{size:>5d}]")
            print(f"Classifier loss: {loss.item():>7f}")
            
        return loss.item()