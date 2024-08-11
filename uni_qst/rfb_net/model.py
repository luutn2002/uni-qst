from torch import nn
import torch

class GaussianNoise(nn.Module):
    """Apply additive zero-centered Gaussian noise. This is useful to mitigate overfitting (you could see it as a form of random data augmentation). Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    - Args:
      - std(`float`): Standard deviation of the noise distribution.
      - mean(`float`): Mean of the noise distrbution.
      - device(`str`): Torch device to place module.
    """

    def __init__(self, 
                 std, 
                 mean=0.):
        super().__init__()
        self.mean = mean
        self.std = std
   
    def forward(self, x):
        return x + torch.randn(x.size())*self.std + self.mean
    
class ComplexLinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu_r = nn.LeakyReLU(negative_slope=0.1)

        self.real_mapping = nn.Linear(7, 1024)
        
        self.linear_real = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 3))
        
        self.imag_mapping = nn.Linear(7, 1024)
        
        self.linear_imag = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 3))
        
    def forward(self, x, ft):
        
        real = self.real_mapping(x) + ft
        real = self.leaky_relu_r(real)
        real = self.linear_real(real)

        imag = self.imag_mapping(x) + ft
        imag = self.leaky_relu_r(imag)
        imag = self.linear_imag(imag)
        
        return x, torch.complex(real, imag)

class RFBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.drop_out = nn.Dropout(p=0.4)
        self.noise = GaussianNoise(0.005)
        self.conv3 = nn.Conv2d(32, 64, 3, bias=False, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, bias=False)
        self.conv6 = nn.Conv2d(128, 64, 3, bias=False, stride=2)
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 7)
        )
        
        self.complex_classifier = ComplexLinearRegressor()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.noise(x)
        x = self.drop_out(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.noise(x)
        x = self.drop_out(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        x = self.leaky_relu(x)
        x = self.drop_out(x)
        
        ft = torch.flatten(x, start_dim=1)
        
        x = self.linear(ft)      
        x, complex_vec = self.complex_classifier(x, ft)
        
        return x, complex_vec