
from .Base import BaseLayers

class Dropout(BaseLayers):
    def __init__(self, probability):
        super(Dropout, self).__init__()
        pass

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass