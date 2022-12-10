import numpy as np
from scipy import signal
from .Base import BaseLayers


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

    def forward(self, input_tensor):

        self.b, self.cI, self.y, self.x = input_tensor.shape

        out_height = int((self.y + 2 * self.pad - self.m) / self.stride_y) + 1
        out_width = int((self.x + 2 * self.pad - self.n) / self.stride_x) + 1

        padded = np.pad(input_tensor, [(0,), (0,), (self.pad,), (self.pad,)], 'constant')
        output_tensor = np.zeros((self.b, self.num_kernels, out_height, out_width))


        for batch in range(self.b):
            for c in range(self.num_kernels):
                output_layer = output_tensor[batch][c]
                # iterate over
                for y in range(out_height):
                    for x in range(out_width):
                        kernel = padded[batch, 0, y*self.stride_y:y*self.stride_y+self.m, x*self.stride_x:x*self.stride_x+self.n]
                        weighz = self.weights[c][0]
                        biaz = self.bias[c]
                        dot = np.matmul(kernel, weighz) + biaz
                        sum = np.sum(dot)
                        output_layer[y][x] = sum
                        #output_layer[y][x] = np.sum(np.dot(padded[batch, :, y*self.stride_y:y*self.stride_y+self.m,
                         #                                  x*self.stride_x:x*self.stride_x+self.n]
                                       #                    , self.weights[c])) + self.bias[c]
                        pass
        return output_tensor

    def backward(self, error_tensor):
        return


