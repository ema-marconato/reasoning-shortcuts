# -*- coding: utf-8 -*-
"""
Conceptizers
"""
import sys
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.autograd import Variable

#===============================================================================
#=======================       MODELS FOR IMAGES       =========================
#===============================================================================


"""
class AutoEncoder:
    called by image_*_conceptizers. 
    encoder: c()
    decoder: d() (discriminator)
def __init__:
    initialization 
def forward:
    compute d(c)
"""
class AutoEncoder(nn.Module):

    """
    def __init__:
        initialization
    Input:
        None (except self)
    Return:
        None
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()

    """
    def forward:
        compute d(e(c(x)))
    Input:
        x: e(x)
    Return:
        encoded_1: predicted known concepts
        encoded_2: predicted unknown concepts
        real_fake: discriminator's outout (0 is fake, 1 is real)        
    """
    def forward(self, x):
        
        # compute concepts
        encoded_1, encoded_2 = self.encode(x)

        # resize for following process
        x = x.view(x.size(0), -1)

        
        # compute dicriminator
        real_fake = self.decode(x, encoded_1, encoded_2)

        return encoded_1, encoded_2, real_fake

"""
class image_fcc_conceptizer:
    network definitions of encoder and decoder using fully connected network
    encoder c() is the network by computing concept c(e(x)) 
    decoder d() is the discriminator network
def __init__:
    define parameters (e.g., # of layers) 
def encode:
    compute concepts
def decode:
    compute discriminator's output
"""
class image_fcc_conceptizer(AutoEncoder):
    """ 
    def __init__:
        define parameters (e.g., # of layers) 
        MLP-based conceptizer for concept basis learning.
    Inputs:
        din (int): input size
        nconcept (int): # of all concepts
        nconcept_labeled (int): # of known concepts
        cdim (int): concept dimension
        sparsity (int) : the number of kWTA's hyperparameter
            kWTA: https://github.com/a554b554/kWTA-Activation/
    Return:
        None
    """
    def __init__(self, din, nconcept, nconcept_labeled, cdim, sparsity, senn):
        super(image_fcc_conceptizer, self).__init__()
        
        # set self hyperparameters
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.cdim     = cdim       # Dimension of each concept (default 1)
        self.learnable = True

        self.sparsity = sparsity # Number of kWTA
        
        self.nconcept = nconcept   # Number of all concepts
        self.nconcept_labeled = nconcept_labeled # Number of unknown concepts

        self.senn = senn # flag of senn
        
        """
        encoding
        self.enc1: encoder for known concepts
        self.enc2: encoder for unknown concepts
        """
        if senn == True:
            self.enc = nn.Linear(self.din, self.nconcept)
        else:
            self.enc1 = nn.Linear(self.din, self.nconcept_labeled)
            self.enc2 = nn.Linear(self.din, self.nconcept-self.nconcept_labeled)

        # discriminator (DeepInfoMax; to maximize the mutual information)
        self.dec1 = nn.Linear(self.nconcept+self.din, 512)
        self.bn_dec1 = nn.BatchNorm1d(num_features=512)
        self.dec2 = nn.Linear(512, 512)
        self.bn_dec2 = nn.BatchNorm1d(num_features=1024)
        self.dec3 = nn.Linear(512, 1)

    """ 
    def encode:
        compute concepts
    Inputs:
        x: output of pretrained model (encoder)
    Return:
        encoded_1: predicted known concepts
        encoded_2: predicted unknown concepts
    """
    def encode(self, x):

        # resize
        p = x.view(x.size(0), -1)
        
        if self.senn == True:
            # compute unknown concepts
            encoded = self.enc(p)

            """
            kWTA: https://github.com/a554b554/kWTA-Activation/
            Winner-take-all (WTA) is a computational principle applied in computational models of neural networks 
            by which neurons in a layer compete with each other for activation, which originated decades ago. 
            k-WTA is a natural extension of WTA that retains the k largest values of an input vector 
            and sets all others to be zero before feeding the vector to the next network layer

            By using this activation, it is easy to control the unknown concepts we use to classify
            """
            k = self.sparsity
            topval = encoded.topk(k,dim=1)[0][:,-1]
            topval = topval.expand(encoded.shape[1],encoded.shape[0]).permute(1,0)
            comp = (encoded>=topval).to(encoded)
            encoded = comp*encoded

            # reshape for the following process
            encoded = encoded.reshape([encoded.shape[0],encoded.shape[1],1])
            
            # aling the return's shape
            encoded_1 = encoded
            encoded_2 = encoded
            
        else:
            # compute known concepts, we find thatleaky_relu was the best activation

            #For MNNIST and CUB-200-2011
            #encoded_1 = F.leaky_relu(self.enc1(p))

            #For BDD-OIA
            T = 2.5
            logits_c  = self.enc1(p) / T

            encoded_1 = torch.sigmoid(logits_c)

            

            # compute unknown concepts
            encoded_2 = self.enc2(p)
            
            """
            kWTA: https://github.com/a554b554/kWTA-Activation/
            Winner-take-all (WTA) is a computational principle applied in computational models of neural networks 
            by which neurons in a layer compete with each other for activation, which originated decades ago. 
            k-WTA is a natural extension of WTA that retains the k largest values of an input vector 
            and sets all others to be zero before feeding the vector to the next network layer

            By using this activation, it is easy to control the unknown concepts we use to classify
            """
            k = self.sparsity
            topval = encoded_2.topk(k,dim=1)[0][:,-1]
            topval = topval.expand(encoded_2.shape[1],encoded_2.shape[0]).permute(1,0)
            comp = (encoded_2>=topval).to(encoded_2)
            encoded_2 = comp*encoded_2

            # reshape for the following process
            encoded_2 = encoded_2.reshape([encoded_2.shape[0],encoded_2.shape[1],1])

        return encoded_1, encoded_2

    """ 
    def decode:
        compute discriminator
    Inputs:
        x: output of pretrained model
        z1_list: predicted known concepts
        z2: predicted unknown concepts
    Return:
        p: discriminator's outout (0 is fake, 1 is real) 
    NOTE: 
        this discriminator's architecture is inspired by the global DeepInfoMax model.
        DeepInfoMax@ICLR2019: https://arxiv.org/pdf/1808.06670.pdf
        Their model computes p(X,c(e(X))), where c is concept layer and e is encoder
    """
    def decode(self, x, z1_list, z2):
        
        # reshape for concatenate (known and unknown concepts)
        if self.senn == True:
            z = z2.reshape([z2.shape[0],z2.shape[1]])
        else:
            z2 = z2.reshape([z2.shape[0],z2.shape[1]])
            z  = torch.cat((z1_list,z2),dim=1)

        # concatenate (encoded output and predicted concepts)
        z = torch.cat((x,z),dim=1)

        
        """
        Mish: (out = x*F.tanh(F.softplus(in)))
        https://arxiv.org/abs/1908.08681
        https://github.com/digantamisra98/Mish
        """
        p = self.dec1(z)
        p = p * torch.tanh(F.softplus(p))
        p = self.dec2(p)
        p = p * torch.tanh(F.softplus(p))
        p = self.dec3(p)
        return p

    
"""
class image_fcc_conceptizer:
    network definitions of encoder and decoder using CNN
    encoder c() is the network by computing concept c(e(x)) 
    decoder d() is the discriminator network
def __init__:
    define parameters (e.g., # of layers) 
def encode:
    compute concepts
def decode:
    compute discriminator's output
NOTE: Not maintenance because current version does not use this class, if you want to use CNN-based model, please modify this class.
"""
class image_cnn_conceptizer(AutoEncoder):
    """
    def __init__:
        CNN-based conceptizer for concept basis learning
    Inputs:
        din (int): input size
        nconcept (int): number of concepts
        cdim (int): concept dimension
        nchannel (int) : channel
        sparsity (int) : the number of kWTA's hyperparameter
            kWTA: https://github.com/a554b554/kWTA-Activation/

        NOTE:
            Inputs:
                x: Image (b x c x d x d)
            Output:
                Tensor of encoded concepts (b x nconcept x cdim)
    """
    def __init__(self, din, nconcept, nconcept_labeled, cdim=None, nchannel =1, sparsity = 1):
        super(image_cnn_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.cdim     = cdim       # Dimension of each concept
        self.nchannel = nchannel
        self.learnable = True
        self.add_bias = False
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both

        self.sparsity = sparsity
        
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.nconcept_labeled = nconcept_labeled

        # Encoding
        self.conv1  = nn.Conv2d(nchannel,10, kernel_size=5)    # b, 10, din - (k -1),din - (k -1)
        # after pool layer (functional)                        # b, 10,  (din - (k -1))/2, idem
        self.conv2_1  = nn.Conv2d(10, self.nconcept_labeled, kernel_size=5)   # b, 10, (din - (k -1))/2 - (k-1), idem
        self.conv2_2  = nn.Conv2d(10, self.nconcept-self.nconcept_labeled, kernel_size=5)   # b, 10, (din - (k -1))/2 - (k-1), idem
        self.linear1 = nn.ModuleList() 
        for i in range(self.nconcept_labeled):
            self.linear1.append(nn.Linear(self.dout**2, self.cdim))
        self.linear2 = nn.Linear(self.dout**2, self.cdim)       # b, nconcepts, cdim

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din

    """ 
    def encode:
        compute concepts
    Inputs:
        x: output of pretrained model (encoder)
    Return:
        encoded_1: predicted known concepts
        encoded_2: predicted unknown concepts
    """
    def encode(self, x):
        
        p       = F.relu(F.max_pool2d(self.conv1(x), 2))
        p_1     = F.relu(F.max_pool2d(self.conv2_1(p), 2))
        p_2     = F.relu(F.max_pool2d(self.conv2_2(p), 2))

        # compute known concepts, we find thatleaky_relu was the best activation
        cnt = 0
        encoded_1 = []
        for fc in self.linear1:
            encoded_1.append(fc(p_1.view(-1, self.nconcept_labeled, self.dout**2)[:,cnt]))
            cnt=cnt+1
                             
        # compute unknown concepts
        encoded_2 = self.linear2(p_2.view(-1, self.nconcept-self.nconcept_labeled, self.dout**2))
        encoded_2 = encoded_2.reshape([encoded_2.shape[0],encoded_2.shape[1]])

        """
        kWTA: https://github.com/a554b554/kWTA-Activation/
        Winner-take-all (WTA) is a computational principle applied in computational models of neural networks 
        by which neurons in a layer compete with each other for activation, which originated decades ago. 
        k-WTA is a natural extension of WTA that retains the k largest values of an input vector 
        and sets all others to be zero before feeding the vector to the next network layer
        
        By using this activation, it is easy to control the unknown concepts we use to classify
        """
        k = self.sparsity
        topval = encoded_2.topk(k,dim=1)[0][:,-1]
        topval = topval.expand(encoded_2.shape[1],encoded_2.shape[0]).permute(1,0)
        comp = (encoded_2>=topval).to(encoded_2)
        encoded_2 = comp*encoded_2
        encoded_2 = encoded_2.reshape([encoded_2.shape[0],encoded_2.shape[1],1])
        
        return encoded_1, encoded_2
    
    """ 
    def decode:
        compute discriminator
    Inputs:
        z1_list: predicted known concepts
        z2: predicted unknown concepts
    Return:
        p: discriminator's outout (0 is fake, 1 is real) 
    NOTE: 
        this discriminator's architecture is not the global DeepInfoMax model...
        DeepInfoMax@ICLR2019: https://arxiv.org/pdf/1808.06670.pdf
        Their model computes p(X,c(e(X))), where c is concept layer and e is encoder
        If you use this function, please modify....
    """
    def decode(self, z1_list,z2):
        z1      = torch.cat(z1_list, dim=1)
        z1      = z1.view(-1,self.nconcept_labeled, self.cdim)
        z       = torch.cat((z1,z2),dim=1)
        q       = self.unlinear(z).view(-1, self.nconcept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = torch.tanh(self.deconv1(q))
        return decoded
 
