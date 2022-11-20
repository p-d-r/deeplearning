from .Base import BaseLayers
import numpy as np
BaseLayers()

class ReLU(BaseLayers):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input = input_tensor
        return np.maximum(0.0, input_tensor)

    def backward(self, error_tensor):
        dx = 1 * (self.input >= 0)
        return error_tensor * dx
