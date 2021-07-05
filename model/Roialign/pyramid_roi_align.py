

from torch.autograd.variable import Variable
from model.module import log2
import torch
import math

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    #   Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

        boxes = inputs[0]

        #   Feature Maps. List of feataure maps from different level of the
        #   featrue pyramid. Each is [batch, height, width, channels]
        features_maps = inputs[1:]

        #   Assign each ROI to a level in the pyramid based on the Roi area.
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
        height = y2 - y1
        weight = x2 - x1

        #   Equation 1 in the Feature Pyramid Networks paper. Account for
        #   the fact that our coordinates are normalized here.
        #   e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad = False)
        if boxes.is_cuda:
            image_area = image_area.cuda()
        roi_level = 4 + log2(torch.sqrt(height*weight)/(244.0/torch.sqrt(image_area)))
        roi_level = roi_level.round().int()
        roi_level = roi_level.clamp(2,5)

class RoiAlign_torch(self):
    def __init__(self):
        super().__init__()
        pass