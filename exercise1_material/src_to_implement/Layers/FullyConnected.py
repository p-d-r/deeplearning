from .Base import BaseLayers
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
        self.weights = 0.1*np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self._optimizer = None
        self.batch_size = input_size
        self.gradient_weights = None
        self.gradient_tensor = None


    # forward layer
    def forward(self, input_tensor):

        self.input_tensor = input_tensor

        self.output_forward = np.dot(self.input_tensor, self.weights) + self.biases

        return self.output_forward

    # backward layer
    def backward(self, error_tensor):

        self.output_backward = np.dot(error_tensor, self.weights.T)

        # implement gradient and bias update

        return self.output_backward

    # getter method optimizer
    def get_optimizer(self):
        return self._optimizer

    # setter method optimizer
    def set_optimizer(self, x):
        self._optimizer = x






