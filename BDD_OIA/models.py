# -*- coding: utf-8 -*-
""" 
Detail of forwarding of our model
"""

# -*- coding: utf-8 -*-
import sys
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

# if you set True, many print function is used to debug 
DEBUG = False


#===============================================================================
#====================      SIMPLE FC and CNN MODELS  ===========================
#===============================================================================
"""
class GSENN
    detail of training forwarding of our model
def __init__:
    initialization 
def forward:
    compute out = f( c(e(x)), theta(e(x)) ) 
    where e() is encoder, c() is conceptizer, theta() is parametrizer and f() is aggregator 
"""
class GSENN(nn.Module):
    ''' Wrapper for GSENN with H-learning'''

    """
    def __init__:
        initialization 
    Inputs:
        conceptizer: network to output (known+unknown) concepts
        parametrizer: network to output weights of concepts
        aggregator: aggregation to output predicted task class
        cbm (bool): True if you use CBM model, otherwise False, if True, aggregator f is the linear function
                    not using parametrizer and unknown concepts
    Return:
        None
    """
    def __init__(self, conceptizer, parametrizer, aggregator, cbm, senn):
        super(GSENN, self).__init__()
        self.cbm = cbm
        self.senn = senn
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = conceptizer.learnable

    """
    def forward:
        compute out = f( c(e(x)), theta(e(x)) ) 
        where e() is encoder, c() is conceptizer, theta() is parametrizer and f() is aggregator 
    Inputs:
        x: output of encoder (inception v.3)
    Return:
        out: f( c(e(x)), theta(e(x)) ) or CBM version f( c(e(x)) )
    """
    def forward(self, x):

        if DEBUG:
            print('Input to GSENN:', x.size())

        # Get concepts, h_x_labeled is known, h_x is unknown concepts
        h_x_labeled, h_x, _ = self.conceptizer(x)
        h_x_labeled = h_x_labeled.view(-1,h_x_labeled.shape[1], 1)
        #self.h_norm_l1 = h_x.norm(p=1)

        if DEBUG:
            print('Encoded concepts: ', h_x.size())
            if self.learning_H:
                print('Decoded concepts: ', h_x_labeled.size())

        # Get relevance scores (~thetas)
        thetas = self.parametrizer(x)

        # When theta_i is of dim one, need to add dummy dim
        if len(thetas.size()) == 2:
            thetas = thetas.unsqueeze(2)

        # Store local Parameters
        self.thetas = thetas

        if len(h_x.size()) == 4:
            # Concepts are two-dimensional, so flatten
            h_x = h_x.view(h_x.size(0), h_x.size(1), -1)
        if len(h_x_labeled.size()) == 4:
            h_x_labeled = h_x_labeled.view(h_x_labeled.size(0), h_x.size(1), -1)

        if not self.senn:
            # store (known+unknown) concepts
            self.concepts = torch.cat((h_x_labeled,h_x),dim=1)
            # store known concepts
            self.concepts_labeled = h_x_labeled
        else:
            self.concepts = h_x


        if DEBUG:
            print('Encoded concepts: ', self.concepts.size())
            print('thetas: ', thetas.size())

            
        # if you use cbm, aggregator does not use unknown concepts, even if you define it
        if self.cbm:
            out = self.aggregator(self.concepts_labeled, thetas)
        else:
            out = self.aggregator(self.concepts, thetas)

        return out
