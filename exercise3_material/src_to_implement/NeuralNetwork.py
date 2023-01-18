from Layers import Base
from Optimization import Constraints
import copy

class NeuralNetwork(Base.BaseLayers):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super(NeuralNetwork, self).__init__()

        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer


    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        self.layer_input = self.input_tensor

        loss = 0

        for layer in self.layers:
            self.layer_input = layer.forward(self.layer_input)
            if layer.trainable is True and self.optimizer.regularizer is not None:
                loss = loss + self.optimizer.regularizer.norm(layer.weights)

        self.output = self.loss_layer.forward(self.layer_input, self.label_tensor) + loss

        return self.output

    def backward(self):
        prediction = self.output
        self.loss.append(self.output)
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def get_phase(self):
        return self._testing_phase

    def set_phase(self, testing_phase):
        self._testing_phase = testing_phase

    phase = property(get_phase, set_phase)

    def train(self, iterations):
        self.set_phase(False)
        for i in range(0, iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        self.set_phase(True)
        layer_input = input_tensor
        for layer in self.layers:
            layer.testing_phase = True
            layer_input = layer.forward(layer_input)
        return layer_input

