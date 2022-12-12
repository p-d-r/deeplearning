from .Base import BaseLayers
from .Initializers import Constant, UniformRandom, He, Xavier
import numpy as np
import sys
sys.path.append("..")
from src_to_implement.Optimization import Optimizers


class Conv(BaseLayers):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape  # singular value or tuple
        self.convolution_shape = convolution_shape  # [c,m] or [c,m,n]
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(0, 1, (num_kernels,))
        self.channels, self.m, self.n = convolution_shape
        self.weights = np.random.uniform(0, 1, (num_kernels, self.channels, self.m, self.n))
        self.stride_y, self.stride_x = stride_shape
        self.pad = int(self.m / 2)
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):

        self.b, self.cI, self.y, self.x = input_tensor.shape

        out_height = int((self.y + 2 * self.pad - self.m) / self.stride_y) + 1
        out_width = int((self.x + 2 * self.pad - self.n) / self.stride_x) + 1

        padded = np.pad(input_tensor, [(0,), (0,), (self.pad,), (self.pad,)], 'constant')
        output_tensor = np.zeros((self.b, self.num_kernels, out_height, out_width))

        for batch in range(self.b):
            for kernel in range(self.num_kernels):
                for y in range(out_height):
                    for x in range(out_width):
                        for channels in range(self.cI):
                            kernel_scope = padded[batch,:,y*self.stride_y:y*self.stride_y+self.m, x*self.stride_x:x*self.stride_x+self.n]
                            res = np.sum(kernel_scope * self.weights[kernel]) + self.bias[kernel]
                            output_tensor[batch, kernel, y, x] = res

        return output_tensor

    def backward(self, error_tensor):
        return

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer
        self.weights = weights.initialize(np.shape(self.weights), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        bias = bias_initializer
        self.bias = bias.initialize(np.shape(self.bias), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        return
