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
        layer_input = self.input_tensor
        for layer in self.layers:
            layer_input = layer.forward(layer_input)

        self.output = layer_input
        return self.output

    def backward(self):
        prediction = self.output
        self.loss.append(self.loss_layer.forward(prediction, self.label_tensor))
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
        return

    def train(self, iterations):
        for i in range(0, iterations):
            prediction = self.forward()
            self.backward()

    def test(self, input_tensor):
        layer_input = input_tensor
        for layer in self.layers:
            layer_input = layer.forward(layer_input)

        return layer_input
