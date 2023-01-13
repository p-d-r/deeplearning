from .Base import BaseLayers
from .FullyConnected import FullyConnected
import numpy as np

class BatchNormalization(BaseLayers):
    def __init__(self, channels):
        super(BatchNormalization, self).__init__()
        self.trainable = True
        self.weights = None
        self.channels = channels
        self.epsilon = np.finfo(float).eps
        self.tilde_mean = 0
        self.tilde_variance = 0
        self.k = 0
        self.alpha = 0.8
        self.sigma = 0
        self.shape_0 = 0
        self.shape_1 = 0
        self.shape_2 = 0
        self.shape_3 = 0

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):

        if len(tensor.shape) == 4:
            self.shape_0 = tensor.shape[0]
            self.shape_1 = tensor.shape[1]
            self.shape_2 = tensor.shape[2]
            self.shape_3 = tensor.shape[3]

            tensor_conv = np.zeros((tensor.shape[0], tensor.shape[2] * tensor.shape[3], tensor.shape[1]))

            for i in range(self.shape_0):
                tensor_re = tensor[i]
                tensor_re = np.reshape(tensor_re, (tensor_re.shape[0], tensor_re.shape[1] * tensor_re.shape[2]))
                tensor_re = np.transpose(tensor_re)
                tensor_conv[i] = tensor_re

            tensor_conv = np.reshape(tensor_conv, (tensor_conv.shape[0] * tensor_conv.shape[1], tensor_conv.shape[2]))

        if len(tensor.shape) == 2:

            tensor_conv = np.zeros((self.shape_0, self.shape_1, self.shape_2, self.shape_3))
            tensor = np.reshape(tensor, (self.shape_0, self.shape_2 * self.shape_3, self.shape_1))

            for i in range(self.shape_0):
                tensor_re = tensor[i]
                tensor_re = np.transpose(tensor_re)
                tensor_re = np.reshape(tensor_re, (self.shape_1, self.shape_2, self.shape_3))
                tensor_conv[i] = tensor_re

        return tensor_conv

    def forward(self, input_tensor):

        dim_4 = False
        if len(input_tensor.shape) == 4:
            dim_4 = True
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor

        self.initialize()
        self.mean = np.mean(input_tensor, axis=0)
        self.variance = np.var(input_tensor, axis=0)
        output = (input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)

        self.tilde_mean = self.alpha * self.tilde_mean + (1 - self.alpha) * self.mean
        self.tilde_variance = self.alpha * self.tilde_variance + (1 - self.alpha) * self.variance

        if self.testing_phase == True:
            output = (input_tensor - self.tilde_mean) / np.sqrt(self.tilde_variance + self.epsilon)

        self.output = self.weights * output + self.bias

        if dim_4 == True:
            self.output = self.reformat(self.output)

        return self.output

    def backward(self, error_tensor):

        gradient_weights = np.sum(error_tensor * self.weights)
        gradient_bias = np.sum(self.bias)

        norm_mean = self.input_tensor - self.mean
        var_eps = self.variance + self.epsilon

        gamma_err = error_tensor * self.weights
        inv_batch = 1. / error_tensor.shape[0]

        grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

        sqrt_var = np.sqrt(var_eps)
        first = gamma_err * 1. / sqrt_var

        grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
        grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

        second = grad_var * (2. * norm_mean) * inv_batch
        grad_mu = grad_mu_two + grad_mu_one

        gradient_input = first + second + inv_batch * grad_mu

        return gradient_input, gradient_weights, gradient_bias