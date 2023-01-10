from .Base import BaseLayers

class BatchNormalization(BaseLayers):
    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.trainable = True

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass