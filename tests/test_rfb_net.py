import pytest
import torch
from uni_qst.rfb_net.model import RFBNet
from uni_qst.loss import LinearCombinationLoss
from uni_qst.dataset import CustomTensorDataset, DatasetGenerator
from torch.utils.data import DataLoader
#from uni_qst.rfb_net.ops import BatchedFidelity, Reconstructor
from uni_qst.rfb_net.utils import train_step

def test_rfb_net(batch_size = 2,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        try:
                with device:
                        generator = DatasetGenerator()
                        train_data, train_label, train_value, train_ft, _, _ , _, _ = generator.generate(dataset_size=10, to_numpy = False, save_file = False)

                        train_dataset = CustomTensorDataset((train_data, train_label, train_value, train_ft))

                        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

                        loss = LinearCombinationLoss().to(device)
                        #fidel_score = BatchedFidelity().to(device)
                        model = RFBNet()
                        model.double().to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                        #recon = Reconstructor().to(device)
                        
                        train_step(loss, model, train_dataloader, optimizer, device)
        except Exception as e:
                pytest.fail(f"{e}")