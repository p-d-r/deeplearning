from .Base import BaseLayers
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid
import numpy as np


class RNN(BaseLayers):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_length = input_size + hidden_size
        self.layer1 = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH()
        self.layer2 = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self._optimizer = None
        self.h = None
        self._gradient_weights = np.zeros(self.weights.shape)
        self.gradient_weights2 = np.zeros(self.layer2.weights.shape)
        self.inputs = None
        self.trainable = True
        self.input_tensor = None
        self.h_memo = np.zeros(self.hidden_size)
        self.y = None
        self.bias = None
        self._memorize = False
        self.input_bias = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        # self.layer1.optimizer = value
        # self.layer2.optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def weights(self):
        return self.layer1.weights

    @weights.setter
    def weights(self, value):
        self.layer1.weights = value

    @property
    def bias(self):
        return self.layer1.bias

    @bias.setter
    def bias(self, value):
        self.layer1.bias = value

    def forward(self, input_tensor):
        # hidden_size = number of hidden states
        # -> hidden_size + 2(input-layer, output-layer) == input_tensor.shape[0]?
        self.inputs = np.zeros((input_tensor.shape[0], 1, self.weight_length))
        self.inputs_bias = np.zeros((input_tensor.shape[0], 1, self.weight_length + 1))
        self.h = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        self.h_bias = np.zeros((input_tensor.shape[0], 1, self.hidden_size + 1))
        self.y = np.zeros((input_tensor.shape[0], self.output_size))
        self.input_tensor = input_tensor
        # compute the first run with h_memo = [0,0,0...] or memorized h[-1]
        self.inputs[0] = np.hstack((input_tensor[0], self.h_memo))
        self.h[0][0] = self.tanh.forward(self.layer1.forward(self.inputs[0]))
        self.inputs_bias[0] = self.layer1.input_tensor
        self.y[0] = self.sigmoid.forward(self.layer2.forward(self.h[0]))
        # compute h_i-1 -> h_i, y_i
        for i in range(1, self.input_tensor.shape[0]):
            # convert input vector to matrix for technical purpose
            self.inputs[i] = np.hstack((input_tensor[i], self.h[i - 1][0]))
            self.h[i][0] = self.tanh.forward(self.layer1.forward(self.inputs[i]))
            self.inputs_bias[i] = self.layer1.input_tensor
            self.y[i] = self.sigmoid.forward(self.layer2.forward(self.h[i]))
            self.h_bias[i][0] = self.layer2.input_tensor

        if self.memorize:
            self.h_memo = self.h[-1][0]

        return self.y

    def backward(self, error_tensor):
        # print('backward')
        # 1st h_gradient = 0
        gradients = np.empty(self.input_tensor.shape)
        h_gradient = 0
        h_copy = self.h.copy()
        h_copy[-1] = 0
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_weights2 = np.zeros(self.layer2.weights.shape)
        for i in range(self.input_tensor.shape[0] - 1, -1, -1):
            # set activation of the sigmoid layer accordingly
            self.sigmoid.activation = self.y[i].reshape((1, len(self.y[i])))
            sigmoid_gradient = self.sigmoid.backward(error_tensor[i])
            # set input h_i for the 2nd fully connected layer accordingly
            self.layer2.input_tensor = self.h_bias[i]
            gradient2 = self.layer2.backward(sigmoid_gradient)
            self.gradient_weights2 += self.layer2.gradient_weights.copy()
            if self.optimizer is not None:
                self.layer2.weights = self.optimizer.calculate_update(self.layer2.weights, self.layer2.gradient_weights)
            # sum up the different gradients to get the derivative of the copy procedure
            combined_gradient = gradient2 + h_gradient
            # set the activation of the tanh layer accordingly
            self.tanh.activation = self.h[i]
            tanh_gradient = self.tanh.backward(combined_gradient)
            # set the input for the 1st fully connected layer accordingly
            self.layer1.input_tensor = self.inputs_bias[i]
            gradient = self.layer1.backward(tanh_gradient)
            self.gradient_weights += self.layer1.gradient_weights.copy()
            if self.optimizer is not None:
                self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            # split the gradient to get the respective gradients for h_i and x_i
            gradients[i] = gradient[0][0:self.input_size]
            h_gradient = gradient[0][self.input_size:]

        return gradients

    def initialize(self, weights_initializer, bias_initializer):
        self.layer1.weights = weights_initializer.initialize(np.shape(self.weights), self.input_size, self.hidden_size)
        self.layer1.bias = bias_initializer.initialize(np.shape(self.bias), self.input_size, self.hidden_size)
        self.layer2.weights = weights_initializer.initialize(np.shape(self.layer2.weights), self.hidden_size,
                                                             self.output_size)
        self.layer2.bias = bias_initializer.initialize(np.shape(self.layer2.bias), self.hidden_size, self.output_size)
