from .Base import BaseLayers
import numpy as np

BaseLayers()

class ReLU(BaseLayers):

    def __init__(self):

        super(ReLU, self).__init__()
        self.trainable = super().trainable

    def forward(self, input_tensor):

        new_input_tensor = np.zeros((input_tensor))

        for i in range(input_tensor):
            if input_tensor[i] > 0:
                new_input_tensor[i] = input_tensor[i]
            else:
                new_input_tensor[i] = 0

        return new_input_tensor

    def backward(self, error_tensor):

        previous_error_tensor = np.zeros((error_tensor))

        for i in range(error_tensor):
            if error_tensor[i] > 0:
                previous_error_tensor[i] = error_tensor[i]
            else:
                previous_error_tensor[i] = 0

        return previous_error_tensor

