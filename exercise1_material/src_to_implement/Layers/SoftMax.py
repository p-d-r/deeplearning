from .Base import BaseLayers
import numpy as np

class SoftMax(BaseLayers):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        s = np.max(input_tensor, axis=1)
        s = s[:, np.newaxis]
        e = np.exp(input_tensor - s)
        div = np.sum(e, axis=1)
        div = div[:, np.newaxis]
        self.input = input_tensor
        return e / div

    def backward(self, error_tensor):
        y = self.forward(self.input)
        output = error_tensor * y
        return y * (error_tensor - np.sum(output, axis=1, keepdims=True))