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
        self.input_tensor = None
        self._optimizer = Optimizers.Sgd(1)
        self.gradient_weights = None
        self.gradient_tensor = None

    # forward layer
    def forward(self, input_tensor):
        # number of biases needed = batch_size = input_tensor.shape[0]
        bias_entries = np.ones((1, input_tensor.shape[0])).T
        self.input_tensor = input_tensor
        # hstack -> inserts column for bias values
        return np.dot(np.hstack((input_tensor, bias_entries)), self.weights)

    # backward layer
    def backward(self, error_tensor):
        output_backward = np.dot(error_tensor, self.weights[:-1].T)
        # implement gradient and bias update
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self.weights = self._optimizer.calculate_update(self.weights[:-1], self.gradient_weights)
        #self.weights[-1] = self._optimizer.calculate_update(self.weights[-1], error_tensor)
        return output_backward

    # getter method optimizer
    def get_optimizer(self):
        return self._optimizer

    # setter method optimizer
    def set_optimizer(self, x):
        self._optimizer = x
