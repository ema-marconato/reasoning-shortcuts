import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import math

__all__ = ['FRCNN','RCNN_global']


class RCNN_global(nn.Module):
    def __init__(self, cfg=None, random_select=False):
        super(RCNN_global, self).__init__()
        
        # #load RCNN
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=0)
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,12)
        # self.checkpoint = torch.load('models/bdd100k_24.pth', map_location='cpu')
        # self.model.load_state_dict(self.checkpoint['model'])
        
        # Layers for global feature
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)
        self.conv_glob1 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu_glob1 = nn.ReLU(inplace=True)
        self.conv_glob2 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu_glob2 = nn.ReLU(inplace=True)

        self.lin_glob=nn.Linear(in_features=3136, out_features=2048,bias=True)
        self.relu_glob=nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu_glob1(self.conv_glob1(x))
        x = self.relu_glob2(self.conv_glob2(x))
        x = self.avgpool_glob(x)
        x = x.flatten(start_dim=1)
        x = self.relu_glob(self.lin_glob(x))
        
        return x
    
