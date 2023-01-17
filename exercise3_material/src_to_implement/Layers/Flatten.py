import numpy as np
from .Base import BaseLayers

class Flatten(BaseLayers):
    def __init__(self):
        super(Flatten, self).__init__()
        self.dim = 0

    def forward(self, input_tensor):

        self.dim = len(input_tensor.shape)
        if self.dim == 4:
            self.a, self.b, self.c, self.d = np.shape(input_tensor)
            output = input_tensor.reshape(self.a, self.b * self.c * self.d)
        else:
            output = input_tensor

        return output

    def backward(self, error_tensor):

        if self.dim ==4:
            output = error_tensor.reshape(self.a, self.b, self.c, self.d)
        else:
            output = error_tensor

        return output