from .Base import BaseLayers
from .Initializers import Constant, UniformRandom, He, Xavier
import numpy as np
import sys
from copy import deepcopy
sys.path.append("..")
import scipy.signal
from src_to_implement.Optimization import Optimizers


class Conv(BaseLayers):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape  # singular value or tuple
        self.kernel_shape = convolution_shape  # [c,m] or [c,m,n]
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(0, 1, (num_kernels,))

        self.c, self.m, self.n = [], [], []
        if len(convolution_shape) == 3:
            self.channels, self.m, self.n = convolution_shape
            self.stride_y, self.stride_x = stride_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, self.channels, self.m, self.n))
        else:
            self.channels, self.m = convolution_shape
            self.n = 1
            self.stride_y = stride_shape[0]
            self.weights = np.random.uniform(0, 1, (num_kernels, self.channels, self.m))

        self.pad = int(self.m / 2)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_bias = deepcopy(value)
        self._optimizer_weights = value
        return

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):

        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 4: # 2D conv
            self.b, self.cI, self.y, self.x = input_tensor.shape

            print(len(input_tensor.shape))

            out_height = int((self.y + 2 * self.pad - self.m) / self.stride_y) + 1
            out_width = int((self.x + 2 * self.pad - self.n) / self.stride_x) + 1

            padded = np.pad(input_tensor, [(0,), (0,), (self.pad,), (self.pad,)], 'constant')
            output_tensor = np.zeros((self.b, self.num_kernels, out_height, out_width))

            for batch in range(self.b):
                for kernel in range(self.num_kernels):
                    for y in range(out_height):
                        for x in range(out_width):
                            for channels in range(self.cI):
                                kernel_scope = padded[batch, :, y * self.stride_y:y * self.stride_y + self.m,
                                               x * self.stride_x:x * self.stride_x + self.n]
                                res = np.sum(kernel_scope * self.weights[kernel]) + self.bias[kernel]
                                output_tensor[batch, kernel, y, x] = res
        else: # 1D conv
            self.b, self.cI, self.y = input_tensor.shape

            print(len(input_tensor.shape))

            out_height = int((self.y + 2 * self.pad - self.m) / self.stride_y) + 1

            padded = np.pad(input_tensor, (self.pad, self.pad), 'constant', constant_values=(0, 0))
            output_tensor = np.zeros((self.b, self.num_kernels, out_height))

            for batch in range(self.b):
                for kernel in range(self.num_kernels):
                    for y in range(out_height):
                        for channels in range(self.cI):
                            kernel_scope = padded[batch, :, y * self.stride_y:y * self.stride_y + self.m]
                            res = np.sum(kernel_scope * self.weights[kernel]) + self.bias[kernel]
                            output_tensor[batch, kernel, y] = res

        return output_tensor


    def backward(self, error_tensor):
        # if 1D: append array to 2D
        conv1d = len(error_tensor.shape) == 3
        if conv1d:
            error_tensor = error_tensor[:, :, :, np.newaxis]
            self.weights = self.weights[:, :, :, np.newaxis]
            self.stride_shape = (*self.stride_shape, 1)
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]

        # parameters
        batch_size, num_kernels = error_tensor.shape[:2]
        batch, channel, height, width = np.shape(self.input_tensor)
        error_tensor_expand = np.zeros((batch_size, num_kernels, height, width))
        sh, sw = self.stride_shape

        # expand tensor according to stride shape
        for b in range(batch_size):
            for n in range(num_kernels):
                error_tensor_expand[b, n, ::sh, ::sw] = error_tensor[b, n, :, :]

        # calculate convolution
        output_tensor = np.zeros((batch_size, channel, height, width))
        for b in range(batch_size):
            for c in range(channel):
                conv = []
                for n in range(self.num_kernels):
                    conv.append(scipy.signal.convolve(error_tensor_expand[b, n], self.weights[n, c], mode='same'))

                    output_tensor[b, c, :, :] = np.sum(np.asarray(conv), axis=0)

        padding = np.pad(self.input_tensor, ((0, 0), (0, 0), (int(self.m / 2), int((self.m - 1) / 2)),
                                             (int(self.n / 2), int((self.n - 1) / 2))))

        # calculate gradient with respect to weights
        self._gradient_weights = np.zeros((batch_size, num_kernels, channel, self.m, self.n))
        for b in range(batch_size):
            for n in range(num_kernels):
                for c in range(channel):
                    self._gradient_weights[b, n, c] = scipy.signal.correlate(padding[b, c], error_tensor_expand[b, n],
                                                                             mode='valid')
        self._gradient_weights = np.sum(self._gradient_weights, axis=0)

        # calculate gradient with respect to bias
        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # update weights matrix and bias vector
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        # if 1D: back to its original shape, remove the 4th axis
        if conv1d:
            self.weights = np.squeeze(self.weights, axis=3)
            self.stride_shape = self.stride_shape[:-1]
            self.input_tensor = np.squeeze(self.input_tensor, axis=3)
            output_tensor = np.squeeze(output_tensor, axis=3)

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer
        self.weights = weights.initialize(np.shape(self.weights), np.prod(self.kernel_shape), np.prod(self.kernel_shape[1:]) * self.num_kernels)
        bias = bias_initializer
        self.bias = bias.initialize(np.shape(self.bias), np.prod(self.kernel_shape), np.prod(self.kernel_shape[1:]) * self.num_kernels)
        return
