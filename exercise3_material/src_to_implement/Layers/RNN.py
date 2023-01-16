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
        self.weights = self.layer1.weights
        self.bias = self.weights[-1, :]
        self.tanh = TanH()
        self.layer2 = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self.inputs = None
        self.trainable = True
        self.input_tensor = None
        self.time_steps = None
        self.h = None
        self.y = None
        self._memorize = False

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def forward(self, input_tensor):
        # hidden_size = number of hidden states
        # -> hidden_size + 2(input-layer, output-layer) == input_tensor.shape[0]?
        print('input_size: ', self.input_size)
        print('input_tensor: ', input_tensor.shape)
        print('output_size: ', self.output_size)
        print('hidden_size: ', self.hidden_size)
        self.inputs = np.empty((input_tensor.shape[0], 1, self.weight_length))
        self.h = np.empty((input_tensor.shape[0], 1, self.hidden_size))
        self.y = np.empty((input_tensor.shape[0], self.output_size))
        self.input_tensor = input_tensor
        # initialize h_0 with zeroes
        self.h[0][0] = np.zeros(self.hidden_size)

        # compute h_i-1 -> h_i, y_i
        for i in range(0, self.input_tensor.shape[0]-1):
            # convert input vector to matrix for technical purpose
            self.inputs[i] = [np.hstack((self.h[i][0], input_tensor[i])).reshape(self.weight_length)]
            self.h[i+1][0] = self.tanh.forward(self.layer1.forward(self.inputs[i]))
            self.y[i] = self.sigmoid.forward(self.layer2.forward(self.h[i+1]))
           # print('h_{}:'.format(i))
           # print(self.h[i+1][0])
           # print('y_{}:'.format(i))
           # print(self.y[i])

        return self.y

    def backward(self, error_tensor):
        # 1st h_gradient = 0
        print('error shape', error_tensor.shape)
        shape = np.zeros(self.input_tensor.shape)
        gradients = np.empty(self.input_tensor.shape)
        h_gradient = np.zeros(self.hidden_size)
        # h_gradient = np.zeros((1, self.hidden_size))
        for i in range(self.input_tensor.shape[0]-1, 0, -1):
            # sigmoid layer stores inputs in correct order, so no further adjustment of the input is needed
            print('i, h_gradient: ', i, h_gradient)
            sigmoid_gradient = self.sigmoid.backward(error_tensor[i])
            print('sig shape: ', sigmoid_gradient.shape)
            self.layer2.set_input_tensor(self.h[i-1])
            gradient1 = self.layer2.backward(sigmoid_gradient)
            print('gradient1 shape', gradient1.shape)
            combined_gradient = gradient1 + h_gradient
            print('comb shape: ', combined_gradient.shape)
            tanh_gradient = self.tanh.backward(combined_gradient)
            print('tanh shape', tanh_gradient.shape)
            self.layer1.set_input_tensor(self.inputs[i])
            gradient = self.layer1.backward(tanh_gradient)
            print('gradient shape', gradient.shape)
            gradients[i] = gradient[0][0:self.input_size]
            h_gradient = gradient[0][self.input_size:]
            print(len(gradients[i]))
            print(len(h_gradient))

        return gradients

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer
        self.weights = weights.initialize(np.shape(self.weights), self.input_size, self.output_size)
        bias = bias_initializer
        self.bias = bias.initialize(np.shape(self.bias), self.input_size, self.output_size)
