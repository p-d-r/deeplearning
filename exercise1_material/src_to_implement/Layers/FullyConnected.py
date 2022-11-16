from .Base import BaseLayers
import numpy as np
from Optimization import Optimizers
import sys

np.random.seed()


class FullyConnected(BaseLayers):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None
        self.gradient_weights = None
        self.gradient_tensor = None

    # forward layer
    def forward(self, input_tensor):
        # number of biases needed = batch_size = input_tensor.shape[0]
        biases = np.ones((1, input_tensor.shape[0])).T
        return np.dot(np.hstack((input_tensor, biases)), self.weights)

    # backward layer
    def backward(self, error_tensor):
        output_backward = np.dot(error_tensor, self.weights.T)
        # implement gradient and bias update
        return output_backward

    # getter method optimizer
    def get_optimizer(self):
        return self._optimizer

    # setter method optimizer
    def set_optimizer(self, x):
        self._optimizer = x
