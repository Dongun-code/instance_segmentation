import os.path as op
import numpy as np
import math

class Config):
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_SHAPE = np.array(
        [IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3])

    BACKBONE_SHAPES = np.array(
        [[int(math.ceil(IMAGE_SHAPE[0] / stride)),
            int(math.ceil(IMAGE_SHAPE[1] / stride))]]
            for stride in BACKBONE_STRIDES])
    )

