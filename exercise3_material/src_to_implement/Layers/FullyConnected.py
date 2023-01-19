from .Base import BaseLayers
import numpy as np
import sys
sys.path.append("../../../../../../..")



np.random.seed()

class FullyConnected(BaseLayers):

    def __init__(self, input_size, output_size):
        super().__init__()
        self._input_tensor = None
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(input_size + 1, output_size))
        self._gradient_weights = None
        self.bias = self.weights[-1, :]
        self._optimizer = None
        self.input_tensor = None
        self.output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor = np.hstack((self.input_tensor, np.ones((self.input_tensor.shape[0], 1))))
        self.output = np.dot(self.input_tensor, self.weights)
        return self.output

    def backward(self, error_tensor):
        input_error = np.dot(error_tensor, self.weights[0:-1, :].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return input_error

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, x):
        self._optimizer = x

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    optimizer = property(get_optimizer, set_optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer
        self.weights = weights.initialize(np.shape(self.weights), self.input_size, self.output_size)
        bias = bias_initializer
        self.bias = bias.initialize(np.shape(self.bias), self.input_size, self.output_size)


