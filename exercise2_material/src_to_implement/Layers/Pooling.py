import numpy as np
from .Base import BaseLayers
import sys
import math

class Pooling(BaseLayers):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.tile_x = pooling_shape[1]
        self.tile_y = pooling_shape[0]
        self.stride_x = stride_shape[1]
        self.stride_y = stride_shape[0]
        self.input_tensor = None
        self.batch_size = None
        self.out_height = None
        self.out_width = None

    # compute pooling for one given layer
    def forward_pool(self, input_layer):
        out = np.zeros([self.out_height, self.out_width])
        for y in range(0, self.out_height):
            for x in range(0, self.out_width):
                # check if the tile / kernel fits over the input matrix at (x,y)
                limit_y = y * self.stride_y + self.tile_y
                limit_x = x * self.stride_x + self.tile_x

                # compute max of the input_layer with respect to the kernel at position (x,y) of the input
                if limit_y <= input_layer.shape[0] and limit_x <= input_layer.shape[1]:
                    out[y][x] = input_layer[y*self.stride_y:limit_y, x*self.stride_x:limit_x].max()

        return out

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # compute output dimensions
        self.out_height = math.floor(1 + (input_tensor[0][0].shape[0] - self.tile_y) / self.stride_y)
        self.out_width = math.floor(1 + (input_tensor[0][0].shape[1] - self.tile_x) / self.stride_x)
        out = np.zeros((input_tensor.shape[0], input_tensor.shape[1], self.out_height, self.out_width))
        # iterate over batches and channels
        for i in range(0, input_tensor.shape[0]):
            for j in range(0, input_tensor.shape[1]):
                out[i][j] = self.forward_pool(input_tensor[i][j])

        return out

# compute pooling for one given layer
    def backward_pool(self, input_layer, error_tensor, next_error):
        for y in range(0, self.out_height):
            for x in range(0, self.out_width):
                # check if the tile / kernel fits over the input matrix at (x,y)
                limit_y = y * self.stride_y + self.tile_y
                limit_x = x * self.stride_x + self.tile_x
                if limit_y <= input_layer.shape[0] and limit_x <= input_layer.shape[1]:
                    # compute max of the input_layer with respect to the kernel at position (x,y)
                    # of the input and set corresponding value of next_error accordingly
                    next_error[np.where(input_layer == input_layer[y*self.stride_y:limit_y,
                                                                   x*self.stride_x:limit_x].max())] \
                        += error_tensor[y][x]

    def backward(self, error_tensor):
        next_error = np.zeros(self.input_tensor.shape)
        # iterate over all batches and channels
        for i in range(0, self.input_tensor.shape[0]):
            for j in range(0, self.input_tensor.shape[1]):
                self.backward_pool(self.input_tensor[i][j], error_tensor[i][j], next_error[i][j])

        return next_error
