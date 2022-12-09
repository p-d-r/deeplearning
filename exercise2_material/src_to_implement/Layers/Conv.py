import numpy as np
from scipy import signal
from .Base import BaseLayers


class Conv(BaseLayers):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape # singular value or tuple
        self.convolution_shape = convolution_shape # [c,m] or [c,m,n]
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(0, 1, (num_kernels,))
        self.cK, self.m, self.n = convolution_shape
        self.weights = np.random.uniform(0, 1, (num_kernels, self.cK, self.m, self.n))
        self.stride1, self.stride2 = stride_shape
        self.pad = int(self.m / 2)

    def forward(self, input_tensor):

        self.b, self.cI, self.y, self.x = input_tensor.shape

        h_output = int((self.y + 2 * self.pad - self.m) / self.stride1) + 1
        w_output = int((self.x + 2 * self.pad - self.n) / self.stride2) + 1

        padded = np.pad(input_tensor, [(0,), (0,), (self.pad,), (self.pad,)], 'constant')
        output_tensor = np.zeros((self.b, self.cI, h_output, w_output))

        for n in range(self.b):
            for f in range(self.num_kernels):
                for i in range(h_output):
                    for j in range(w_output):
                        output_tensor[n, f, i, j] = np.sum( padded[n, :, i*self.stride1:i*self.stride1+self.m, j*self.stride2:j*self.stride2+self.n] * self.weights[f] ) + self.bias[f]

        return output_tensor

    def backward(self, error_tensor):
        return


