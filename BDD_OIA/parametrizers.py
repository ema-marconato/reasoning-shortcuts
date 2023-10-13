# -*- coding: utf-8 -*-
""" 
Parametrizers
"""

# -*- coding: utf-8 -*-

# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import pdb
import numpy as np


"""
class dfc_parametrizer:
    fully connected-based model parameter theta(x) of SENN
def __init__:
    initialization 
def forward:
    compute theta(x)
NOTE: Based on the SENN's paper, theta(x) should be more complex network than conceptizer
      SENN: https://arxiv.org/abs/1806.07538
"""
class dfc_parametrizer(nn.Module):
    """ 
    def __init__:
        initialization
    Inputs:
        din (int): input concept dimension
        dout (int): output dimension (1 or number of label classes usually)
        hdim1~4 (int): dimension of 1~4 hidden layers
        nconcept (int): # of (known+unknown) concepts
        layers (int): # of layers
    Returns:
        None
    """
    def __init__(self, din, hdim1, hdim2, hdim3, hdim4, nconcept, dout, layers = 2):
        super(dfc_parametrizer, self).__init__()
        
        print(din, hdim1, hdim2, hdim3, hdim4, nconcept, dout)
        
        # redefine
        self.nconcept = nconcept
        self.din = din
        self.dout = dout
        self.layers = layers

        # network
        self.linear1 = nn.Linear(din, hdim1)
        self.bn1 = nn.BatchNorm1d(num_features=hdim1)
        self.linear2 = nn.Linear(hdim1, hdim2)
        self.bn2 = nn.BatchNorm1d(num_features=hdim2)
        
        """
        self.layers = 3, define the parametrizer for aux. output of encoder
        self.layers = 4, define the parametrizer for the final layer of encoder
        """
        if self.layers == 3:
            self.linear3_2 = nn.Linear(hdim2, self.nconcept*self.dout)
        else:
            self.linear3_1 = nn.Linear(hdim2, hdim3)
            self.bn3 = nn.BatchNorm1d(num_features=hdim3)
            self.linear4 = nn.Linear(hdim3, self.nconcept * self.dout)
    
    """ 
    def forward:
        compute theta(x)
    Input:
        x: encoder's output
    Return:
        p: theta(x)
    """
    def forward(self, x):
        p = self.bn1(self.linear1(x))
        p = p * torch.tanh(F.softplus(p))
        p = self.bn2(self.linear2(p))
        p = p * torch.tanh(F.softplus(p))

        """
        self.layers = 3, define the parametrizer for aux. output of encoder
        self.layers = 4, define the parametrizer for the final layer of encoder
        """
        if self.layers == 3:
            p = self.linear3_2(p) 
        else:
            p = self.bn3(self.linear3_1(p))
            p = p * torch.tanh(F.softplus(p))
            p = self.linear4(p) 

        # reshape for after process
        if self.dout > 1:
            p = p.view(p.shape[0], self.nconcept, self.dout)

        return p

"""
class image_parametrizer:
    CNN-based model parameter theta(x) of SENN
def __init__:
    initialization 
def forward:
    compute theta(x)
NOTE1: Based on the SENN's paper, theta(x) should be more complex network than conceptizer
      SENN: https://arxiv.org/abs/1806.07538
NOTE2: [Not maintenance] because current version does not use this class, if you want to use CNN-based model, please modify this class.
"""
class image_parametrizer(nn.Module):
    """ 
    def __init__:
        initialization 
    Inputs:
        din (int): input concept dimension
        dout (int): output dimension (1 or number of label classes usually)
        nconcept (int): the number of (known+unknown) concepts
        nchannel (int): # of input channel
        only_positive (bool): if true, output activation uses sigmoid, otherwise tanh
    Return:
        None
    """
    def __init__(self, din, nconcept, dout, nchannel = 1, only_positive = False):
        super(image_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.dout = dout
        self.din  = din
        self.conv1 = nn.Conv2d(nchannel, 10, kernel_size=5)   # b, 10, din - (k -1), same
        # after ppol layer with stride=2: din/2 - (k -1)/2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)         # b, 20, din/2 - 3(k -1)/2, same
        # after ppol layer with stride=2: din/4 - 3(k -1)/4
        self.dout_conv = int(np.sqrt(din)//4 - 3*(5-1)//4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*(self.dout_conv**2), nconcept*dout)
        self.positive = only_positive

    """ 
    def forward:
        compute theta(x)
    Input:
        x: encoder's output
    Return:
        out: theta(x)
    """
    def forward(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), 2))
        p = p.view(-1, 20*(self.dout_conv**2))
        p = self.fc1(p)
        out = F.dropout(p, training=self.training).view(-1,self.nconcept,self.dout)
        # if self.positive is true, output activation uses sigmoid, otherwise tanh
        if self.positive:
            out = F.sigmoid(out) # For fixed outputdim, sum over concepts = 1
        else:
            out = torch.tanh(out)
        return out
