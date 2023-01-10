
from .Base import BaseLayers
import numpy as np

class Dropout(BaseLayers):
    def __init__(self, probability):
        super(Dropout, self).__init__()
        self.probability = probability # determines fraction units to keep
        pass

    def forward(self, input_tensor):

        if self.testing_phase == False:
            self.binary_value = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability
            res = np.multiply(input_tensor, self.binary_value)
            res /= self.probability  # inverted dropout technique

        else:
            res = input_tensor

        return res

    def backward(self, error_tensor):

        if self.testing_phase == False:
            res = np.multiply(error_tensor, self.binary_value)
            res /= self.probability  # inverted dropout technique

        else:
            res = error_tensor

        return res