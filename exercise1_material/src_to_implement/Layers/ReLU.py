from .Base import BaseLayers
import numpy as np

BaseLayers()


class ReLU(BaseLayers):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.copy()
        output_tensor[output_tensor < 0] = 0
        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = error_tensor.copy()
        previous_error_tensor[previous_error_tensor < 0] = 0
        return previous_error_tensor
