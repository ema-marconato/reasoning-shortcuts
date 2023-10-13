import torch.nn
from torch import nn

from backbones.base.ops import *

class MNISTPairsEncoder(nn.Module):
    def __init__(self, img_channels=1, hidden_channels=32, c_dim=20, latent_dim=20, dropout=0.5):
        super(MNISTPairsEncoder, self).__init__()

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
            padding=1)  # hidden_channels x 14 x 28

        self.enc_block_2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1)  # 2*hidden_channels x 7 x 14

        self.enc_block_3 = nn.Conv2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1)  # 4*hidden_channels x 2 x 7

        self.flatten = Flatten()

        self.dense_c = nn.Linear(
            in_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1],
            out_features=self.c_dim)

        self.dense_mu = nn.Linear(
            in_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1],
            out_features=self.latent_dim)

        self.dense_logvar = nn.Linear(
            in_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1],
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

        # print(x.size())
        
        c, mu, logvar = self.dense_c(x), self.dense_mu(x), self.dense_logvar(x)

        # return encodings for each object involved
        c      = torch.stack(torch.split(c,      self.c_dim//2,      dim=-1), dim=1)
        mu     = torch.stack(torch.split(mu,     self.latent_dim//2, dim=-1), dim=1)
        logvar = torch.stack(torch.split(logvar, self.latent_dim//2, dim=-1), dim=1)

        return c, mu, logvar

class MNISTPairsDecoder(nn.Module):
    def __init__(self, img_channels=1, hidden_channels=32, c_dim=20, latent_dim=20,  dropout=0.5,
                 **params):
        super(MNISTPairsDecoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.padding = [(1, 0), (0, 0), (1, 1)]
        self.unflatten_dim = (3, 7)
        self.c_dim = c_dim

        self.dense = nn.Linear(
            in_features=self.latent_dim + self.c_dim,
            out_features=4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1])

        self.unflatten = UnFlatten()

        self.dec_block_1 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 4,
            out_channels=self.hidden_channels * 2,
            kernel_size=(5, 4),
            stride=2,
            padding=1)

        self.dec_block_2 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        self.dec_block_3 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels,
            out_channels=self.img_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):

        # Unflatten Input
        x = self.dense(x)
        x = self.unflatten(x, self.hidden_channels*4, self.unflatten_dim)

        # MNISTPairsDecoder block 1
        x = self.dec_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 2
        x = self.dec_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsDecoder block 3
        x = self.dec_block_3(x)
        x = torch.nn.Sigmoid()(x)
        return x