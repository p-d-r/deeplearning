from .Base import BaseLayers
import numpy as np


class TanH(BaseLayers):
    def __init__(self):
        super().__init__()
        self.activations = []

    def forward(self, input_tensor):
        self.activations.append((np.exp(input_tensor) - np.exp(-input_tensor)) /
                                (np.exp(input_tensor) + np.exp(-input_tensor)))
        return self.activations[-1]

    def backward(self, error_tensor):
        # error_tensor * f'(input)
        return error_tensor * (1 - self.activations.pop()**2)
