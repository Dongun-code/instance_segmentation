import torch.nn as nn
from model.fpn import FPN
from model.resnet import ResNet
from model.rpn import RPNet
from config import Config as cfg

class MaskRcnn:
    def __init__(self):
        pass

    def structure(self, input_tensor):
        
        #   ResNet
        resnet = ResNet("resnet101",stage5 = True)
        stage = resnet.stage5
        C1, C2, C3, C4, C5 = stage

        #   FPN
        fpn = FPN(C1, C2, C3, C4, C5, out_channels = 256)

        #   Generate Anchors


        #   RPN
        rpn = RPNet(len(cfg.RPN_ANCHOR_RATIOS), cfg.RPN_ANCHOR_STRIDE, 256)

