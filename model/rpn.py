import torch.nn as nn
import torch.nn.functional as F
from model.module import SamePad2d

#   Region Proposal Network
class RPNet(nn.Module):
    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPNet, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth
    
        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stire)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4* anchors_per_location, kernel_size=1, stride=1)
        
    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        #   Anchor score. [ batch, anchors num * 2, height, width]
        rpn_class_logits = self.conv_class(x)

        #   Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2 ,3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        #   Softmax on last dimension of BG/FG
        rpn_probs = self.softmax(rpn_class_logits)

        #   Boudning box refinement. [batch, H, W, anchors per location, depth]
        #   Hmm [batch, anchros num * 4, height, width] ?
        #   where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contigous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]



class RPNet_torch(nn.Module):
    def __init__(self):
        super().__init__()
        pass