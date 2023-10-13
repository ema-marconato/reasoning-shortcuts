import torch.nn
from torch import nn

from backbones.base.ops import *

class MNISTRepeatedEncoder(nn.Module):
    def __init__(self, img_channels=1, hidden_channels=32, c_dim=10, latent_dim=10, dropout=0.5):
        super(MNISTRepeatedEncoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim

        self.unflatten_dim = (3, 7)
        
        self.enc_block_11 = nn.Conv2d(
                                    in_channels=self.img_channels,
                                    out_channels=self.hidden_channels,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        self.enc_block_12 = nn.Conv2d(
                                    in_channels=self.img_channels,
                                    out_channels=self.hidden_channels,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        
        self.enc_block_21 = nn.Conv2d(
                                    in_channels=self.hidden_channels,
                                    out_channels=self.hidden_channels * 2,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        
        self.enc_block_22 = nn.Conv2d(
                                    in_channels=self.hidden_channels,
                                    out_channels=self.hidden_channels * 2,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        
        self.enc_block_31 = nn.Conv2d(
                                    in_channels=self.hidden_channels * 2,
                                    out_channels=self.hidden_channels * 4,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
        
        self.enc_block_32 = nn.Conv2d(
                                    in_channels=self.hidden_channels * 2,
                                    out_channels=self.hidden_channels * 4,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)

        self.flatten = Flatten()

        self.dense_c1 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.c_dim)

        self.dense_mu1 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)

        self.dense_logvar1 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)

        self.dense_c2 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.c_dim)

        self.dense_mu2 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)

        self.dense_logvar2 = nn.Linear(in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
                                 out_features=self.latent_dim)


        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward1(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_11(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_21(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_31(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        # print(x.size())
        
        c, mu, logvar = self.dense_c1(x), self.dense_mu1(x), self.dense_logvar1(x)
        
        return c, mu, logvar
    
    def forward2(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_12(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_22(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_32(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        # print(x.size())
        
        c, mu, logvar = self.dense_c2(x), self.dense_mu2(x), self.dense_logvar2(x)
        
        return c, mu, logvar
        

    def forward(self, x):
        
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        
        c1, mu1, logvar1 = self.forward1(xs[0])
        c2, mu2, logvar2 = self.forward1(xs[1])

        # return encodings for each object involved
        c      = torch.stack((c1, c2), dim=1)
        mu     = torch.stack((mu1, mu2), dim=1)
        logvar = torch.stack((logvar1, logvar2), dim=1)

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