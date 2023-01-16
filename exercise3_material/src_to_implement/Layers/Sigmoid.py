from .Base import BaseLayers
import numpy as np


class Sigmoid(BaseLayers):
    def __init__(self):
        super().__init__()
        self.activations = []

    def forward(self, input_tensor):
        self.activations.append(1/(1+np.exp(-input_tensor)))
        return self.activations[-1]

    def backward(self, error_tensor):
        # error_tensor * f'(input_tensor)
        activation = self.activations.pop()
        return error_tensor * (activation * (1 - activation))


