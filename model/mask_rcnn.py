import torch.nn as nn
from model.fpn import FPN
from model.resnet import ResNet
from model.rpn import RPNet
from config import Config as cfg
from torch.autograd import Variable
from model.module import generate_anchors
import torch

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
        self.anchor = Variable(torch.from_numpy(generate_anchors(cfg.RPN_ANCHOR_SCALES,
                                                                 cfg.RPN_ANCHOR_RATIOS,
                                                                 cfg.BACKBONE_SHAPES,
                                                                 cfg.BACKBONE_STRIDES)).float(), requires_grad=False)

        self.anchors = self.anchors.cuda()                                                                 
        #   RPN
        rpn = RPNet(len(cfg.RPN_ANCHOR_RATIOS), cfg.RPN_ANCHOR_STRIDE, 256)

