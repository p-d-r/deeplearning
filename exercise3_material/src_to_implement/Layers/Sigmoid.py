from .Base import BaseLayers
import numpy as np


class Sigmoid(BaseLayers):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = (1/(1+np.exp(-input_tensor)))
        return self.activation

    def backward(self, error_tensor):
        # error_tensor * f'(input_tensor)
        return error_tensor * (self.activation * (1 - self.activation))


