from .Base import BaseLayers
import numpy as np


class TanH(BaseLayers):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = (np.exp(input_tensor) - np.exp(-input_tensor)) / \
                         (np.exp(input_tensor) + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # error_tensor * f'(input)
        return error_tensor * (1 - self.activation**2)
