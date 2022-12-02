import numpy as np
from .Base import BaseLayers


class Conv(BaseLayers):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

    def forward(self, input_tensor):
        return

    def backward(self, error_tensor):
        return


