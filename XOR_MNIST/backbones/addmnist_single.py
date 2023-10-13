import torch.nn
from torch import nn
from backbones.base.ops import *

class MNISTSingleEncoder(nn.Module):
    def __init__(self, img_channels=1, hidden_channels=32, c_dim=10, latent_dim=10, dropout=0.5):
        super(MNISTSingleEncoder, self).__init__()

        self.channels=3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim 
    
        self.unflatten_dim = (3, 7)
        
        self.enc_block_1 = nn.Conv2d(
                                    in_channels=self.img_channels,
                                    out_channels=self.hidden_channels,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)

        self.enc_block_2 = nn.Conv2d(
                                    in_channels=self.hidden_channels,
                                    out_channels=self.hidden_channels * 2,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        
        self.enc_block_3 = nn.Conv2d(
                                    in_channels=self.hidden_channels * 2,
                                    out_channels=self.hidden_channels * 4,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)

        self.flatten = Flatten()

        self.dense_c = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.c_dim)

        self.dense_mu = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)

        self.dense_logvar = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3
        c, mu, logvar = self.dense_c(x), self.dense_mu(x), self.dense_logvar(x)

        # return encodings for each object involved
        c      = torch.stack(torch.split(c,      self.c_dim,      dim=-1), dim=1)
        mu     = torch.stack(torch.split(mu,     self.latent_dim, dim=-1), dim=1)
        logvar = torch.stack(torch.split(logvar, self.latent_dim, dim=-1), dim=1)

        return c, mu, logvar