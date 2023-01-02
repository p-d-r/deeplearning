from .Base import BaseLayers
import numpy as np


class TanH(BaseLayers):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.output = (np.exp(input_tensor) - np.exp(-input_tensor)) / (np.exp(input_tensor) + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        # error_tensor * f'(input)
        return error_tensor * (1 - self.output**2)
