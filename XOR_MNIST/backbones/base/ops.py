from torch import nn
import torch

class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        s = [x.size(0)]
        s += self.size
        x = x.view(s)
        return x

class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input,start_dim=1)


class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.reshape(input.size(0), hidden_channels, dim[0], dim[1])