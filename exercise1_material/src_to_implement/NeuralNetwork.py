import numpy as np
from Layers import FullyConnected, ReLU, SoftMax, Base
from Optimization import Loss, Optimizers
import copy

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()

        for i in self.data_layer:
            if i == 0:
                self.output = self.data_layer[i].forward(self.input_tensor)
            else:
                self.output = self.data_layer[i].forward(self.output)

        return self.output

    def backward(self):
        return

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
        return

    def train(self, iterations):
        self.loss.append(self.loss)
        return

    def test(self, input_tensor):

        prediction = None

        return prediction
