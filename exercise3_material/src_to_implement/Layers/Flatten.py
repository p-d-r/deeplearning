import numpy as np
from .Base import BaseLayers

class Flatten(BaseLayers):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):

        self.a, self.b, self.c, self.d = np.shape(input_tensor)
        output = input_tensor.reshape(self.a, self.b * self.c * self.d)

        return output

    def backward(self, error_tensor):

        output = error_tensor.reshape(self.a, self.b, self.c, self.d)

        return output