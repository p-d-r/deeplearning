import numpy as np
from .Base import BaseLayers

class Pooling(BaseLayers):
    def __init__(self, stride_shape, pooling_shape):
        super(BaseLayers, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        return

    def backward(self, error_tensor):
        return