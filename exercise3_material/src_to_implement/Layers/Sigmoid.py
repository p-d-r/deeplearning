from .Base import BaseLayers
import numpy as np


class Sigmoid(BaseLayers):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.output = 1/(1+np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        # error_tensor * f'(input_tensor)
        return error_tensor * (self.output * (1 - self.output))


