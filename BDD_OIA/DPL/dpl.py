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


def expand_normalize_concepts(concepts: torch.Tensor):

    assert len(concepts[concepts <0]) == 0 and len(concepts[concepts >1]) == 0, concepts[:10,:,0] 
    
    pC = []
    for i in range(concepts.size(1)):
        # add offset
        c = torch.cat((1- concepts[:, i], concepts[:, i]), dim=1) + 1e-5
        with torch.no_grad():
            Z = torch.sum(c, dim=1, keepdim=True)
        pC.append(c / Z)
    pC = torch.cat(pC, dim=1)

    return pC

def create_w_to_y():
    three_bits_or = torch.cat( (torch.zeros((8,1)), torch.ones((8,1))), dim=1).to(dtype=torch.float) 
    three_bits_or[0] = torch.tensor([1,0]) 

    six_bits_or = torch.cat((torch.zeros((64,1)), torch.ones((64,1))), dim=1).to(dtype=torch.float) 
    six_bits_or[0] = torch.tensor([1,0]) 

    
    and_not_for_stop = torch.tensor([[0,1], 
                                     [0,1],
                                     [1,0],
                                     [0,1]], dtype=torch.float)
    
    and_not = torch.tensor([[0],
                            [0],
                            [1],
                            [0]], dtype=torch.float)

    return three_bits_or, six_bits_or, and_not_for_stop, and_not


class DPL(nn.Module):
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
    def __init__(self, conceptizer, parametrizer, aggregator, cbm, senn, device):
        super(DPL, self).__init__()
        self.cbm = cbm
        self.senn = senn
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = conceptizer.learnable
        self.device = device

        logics = create_w_to_y()

        self.or_three_bits = logics[0].to(self.device)
        self.or_six_bits   = logics[1].to(self.device)
        self.rule_for_stop = logics[2].to(self.device)
        self.rule_lr_move  = logics[3].to(self.device)

        self.pred_5 = nn.Linear(21, 1)
        
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


        if not self.senn:
            # store (known+unknown) concepts
            self.concepts = torch.cat((h_x_labeled,h_x),dim=1)
            # store known concepts
            self.concepts_labeled = h_x_labeled
        else:
            self.concepts = h_x


        if DEBUG:
            print('Encoded concepts: ', self.concepts_labeled.size())

            
        # if you use cbm, aggregator does not use unknown concepts, even if you define it
        out = self.proglob_pred()
        
        return out

    def compute_logic_forward(self, concepts:torch.Tensor): 
        A = concepts[:,  :2].unsqueeze(2).unsqueeze(3)
        B = concepts[:, 2:4].unsqueeze(1).unsqueeze(3)
        C = concepts[:, 4:6].unsqueeze(1).unsqueeze(2)

        poss_worlds = A.multiply(B).multiply(C).view(-1, 2*2*2)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_three_bits)

        # assert torch.abs(active.sum() / len(active)- 1) < 0.001, (active, active.sum() / len(active) )

        return active
    
    def compute_logic_stop(self, concepts:torch.Tensor): 
        A = concepts[:,  6:8 ].unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        B = concepts[:,  8:10].unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        C = concepts[:, 10:12].unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        D = concepts[:, 12:14].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        E = concepts[:, 14:16].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        F = concepts[:, 16:18].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        poss_worlds = A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_six_bits)

        return active
    
    def compute_logic_left(self, concepts:torch.Tensor): 
        A = concepts[:, 18:20].unsqueeze(2).unsqueeze(3)
        B = concepts[:, 20:22].unsqueeze(1).unsqueeze(3)
        C = concepts[:, 22:24].unsqueeze(1).unsqueeze(2)
        
        poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_three_bits)

        return active
    
    def compute_logic_no_left(self, concepts:torch.Tensor): 
        A = concepts[:, 24:26].unsqueeze(2).unsqueeze(3)
        B = concepts[:, 26:28].unsqueeze(1).unsqueeze(3)
        C = concepts[:, 28:30].unsqueeze(1).unsqueeze(2)

        poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_three_bits)

        return active
    
    def compute_logic_right(self, concepts:torch.Tensor): 
        A = concepts[:, 30:32].unsqueeze(2).unsqueeze(3)
        B = concepts[:, 32:34].unsqueeze(1).unsqueeze(3)
        C = concepts[:, 34:36].unsqueeze(1).unsqueeze(2)

        poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_three_bits)

        return active
    
    
    def compute_logic_no_right(self, concepts:torch.Tensor): 
        A = concepts[:, 36:38].unsqueeze(2).unsqueeze(3)
        B = concepts[:, 38:40].unsqueeze(1).unsqueeze(3)
        C = concepts[:, 40:42].unsqueeze(1).unsqueeze(2)

        poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

        active = torch.einsum('bi,ik->bk', poss_worlds, self.or_three_bits)

        return active
    
    def proglob_pred(self):
        pC = expand_normalize_concepts(self.concepts_labeled)

        # evaluate if one of them is active
        F_pred  = self.compute_logic_forward(pC)
        S_pred  = self.compute_logic_stop(pC)
        #
        L_pred  = self.compute_logic_left(pC)
        NL_pred = self.compute_logic_no_left(pC)
        #
        R_pred  = self.compute_logic_right(pC)
        NR_pred = self.compute_logic_no_right(pC)


        F_pred = F_pred[...,None]
        S_pred = S_pred[:,None, :]
        w_FS = F_pred.multiply(S_pred).view(-1, 4)
        labels_01 = torch.einsum('bi,ik->bk', w_FS, self.rule_for_stop) 

        L_pred  = L_pred[...,None]
        NL_pred = NL_pred[:,None, :]
        w_L = L_pred.multiply(NL_pred).view(-1, 4)
        labels_2 = torch.einsum('bi,il->bl', w_L, self.rule_lr_move) 

        R_pred  = R_pred[...,None]
        NR_pred = NR_pred[:,None, :]

        w_R = R_pred.multiply(NR_pred).view(-1, 4)
        labels_3 = torch.einsum('bi,il->bl', w_R, self.rule_lr_move)

        labels_4 = torch.sigmoid(self.pred_5(self.concepts_labeled[:, :,0])).view(-1,1)

        pred = torch.cat([labels_01, labels_2, labels_3, labels_4], dim=1) 

        # avoid overflow
        pred = (pred + 1e-5) / (1+2*1e-5) 
        

        return pred
    
    




        

