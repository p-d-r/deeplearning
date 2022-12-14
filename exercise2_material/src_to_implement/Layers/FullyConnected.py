from .Base import BaseLayers
from .Initializers import Constant, UniformRandom, He, Xavier
import numpy as np
import sys
sys.path.append("..")
from src_to_implement.Optimization import Optimizers


np.random.seed()

class FullyConnected(BaseLayers):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        self.bias = self.weights[-1, :]
        self._optimizer = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor = np.append(self.input_tensor, np.ones((self.input_tensor.shape[0], 1)), axis=1)
        self.output = np.dot(self.input_tensor, self.weights)
        return self.output

    def backward(self, error_tensor):
        input_error = np.dot(error_tensor, self.weights[0:-1, :].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer != 0:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return input_error

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, x):
        self._optimizer = x

    def get_gradient_weights(self):
        return self.gradient_weights

    optimizer = property(get_optimizer, set_optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer
        self.weights = weights.initialize(np.shape(self.weights), self.input_size, self.output_size)
        bias = bias_initializer
        self.bias = bias.initialize(np.shape(self.bias), self.input_size, self.output_size)
        return


