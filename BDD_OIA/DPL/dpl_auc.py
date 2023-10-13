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
from DPL.utils_problog import build_world_queries_matrix_FS, build_world_queries_matrix_L, build_world_queries_matrix_R
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
    
    four_bits_or = torch.cat( (torch.zeros((16,1)), torch.ones((16,1))), dim=1).to(dtype=torch.float) 
    four_bits_or[0] = torch.tensor([1,0]) 

    return four_bits_or

class DPL_AUC(nn.Module):
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
        super(DPL_AUC, self).__init__()
        self.cbm = cbm
        self.senn = senn
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = conceptizer.learnable
        self.device = device

        logics = create_w_to_y()

        self.or_four_bits = logics.to(self.device)

        self.FS_w_q = build_world_queries_matrix_FS().to(self.device)
        # self.LR_w_q = build_world_queries_matrix_LR().to(self.device)
        self.L_w_q = build_world_queries_matrix_L().to(self.device)
        self.R_w_q = build_world_queries_matrix_R().to(self.device)

        self.pred_5 = nn.Linear(21, 2)
        
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

    def compute_logic_no_left_lane(self):
        A = self.pC[:, 18:20].unsqueeze(2) # no lane on left
        B = self.pC[:, 22:24].unsqueeze(1) # solid line on left

        obs_worlds = A.multiply(B).view(-1,4)

        no_left_lane = torch.einsum('bi,ik->bk', obs_worlds, self.or_two_bits)
        
        return no_left_lane
    
    def compute_logic_obstacle(self):

        o_car    = self.pC[:, 10:12].unsqueeze(2).unsqueeze(3).unsqueeze(4) #car
        o_person = self.pC[:, 12:14].unsqueeze(1).unsqueeze(3).unsqueeze(4) #person
        o_rider  = self.pC[:, 14:16].unsqueeze(1).unsqueeze(2).unsqueeze(4) #rider
        o_other  = self.pC[:, 16:18].unsqueeze(1).unsqueeze(2).unsqueeze(3) #other obstacle

        obs_worlds = o_car.multiply(o_person).multiply(o_rider).multiply(o_other).view(-1, 16)

        obs_active = torch.einsum('bi,ik->bk', obs_worlds, self.or_four_bits)

        return obs_active
    


    def proglob_pred(self):
        self.pC = expand_normalize_concepts(self.concepts_labeled)
        
        # for forward
        tl_green = self.pC[:,  :2] # traffic light is green
        follow   = self.pC[:, 2:4] # follow car ahead
        clear    = self.pC[:, 4:6] # road is clear
        
        # for stop 
        tl_red = self.pC[:, 6:8]   # traffic light is red 
        t_sign = self.pC[:, 8:10]  # traffic sign present
        obs    = self.compute_logic_obstacle() # generic obstacle

        A = tl_green.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        B =   follow.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        C =    clear.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6) 
        D =   tl_red.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        E =   t_sign.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        F =      obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_FS = A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)
        #
        labels_FS  = torch.einsum('bi,ik->bk', w_FS, self.FS_w_q)
        ## 
        
        # for LEFT
        left_lane     = self.pC[:, 18:20] # there is LEFT lane
        tl_green_left = self.pC[:, 20:22] # tl green on LEFT
        follow_left   = self.pC[:, 22:24] # follow car going LEFT

        # for LEFT-STOP
        no_left_lane = self.pC[:, 24:26] # no lane on LEFT
        l_obs        = self.pC[:, 26:28] # LEFT obstacle
        left_line    = self.pC[:, 28:30] # solid line on LEFT

        AL =     left_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = tl_green_left.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        CL =   follow_left.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        DL =  no_left_lane.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        EL =         l_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL =     left_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_L = AL.multiply(BL).multiply(CL).multiply(DL).multiply(EL).multiply(FL).view(-1, 64)

        label_L = torch.einsum('bi,ik->bk', w_L, self.L_w_q)
        ## 

        # for RIGHT
        rigt_lane     = self.pC[:, 30:32] # there is RIGHT lane
        tl_green_rigt = self.pC[:, 32:34] # tl green on RIGHT
        follow_rigt   = self.pC[:, 34:36] # follow car going RIGHT

        # for RIGHT-STOP
        no_rigt_lane = self.pC[:, 36:38] # no lane on RIGHT
        r_obs        = self.pC[:, 38:40] # RIGHT obstacle
        rigt_line    = self.pC[:, 40:42] # solid line on RIGHT

        AL =     rigt_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = tl_green_rigt.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        CL =   follow_rigt.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        DL =  no_rigt_lane.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        EL =         r_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL =     rigt_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_R = AL.multiply(BL).multiply(CL).multiply(DL).multiply(EL).multiply(FL).view(-1, 64)

        label_R = torch.einsum('bi,ik->bk', w_R, self.R_w_q)

        pred = torch.cat([labels_FS, label_L, label_R], dim=1)  # this is 8 dim

        # avoid overflow
        pred = (pred + 1e-5) / (1+2*1e-5) 
        
        return pred