import numpy as np
np.random.seed()

class Constant():
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value
        self.output = None
    def initialize(self, weights_shape, fan_in, fan_out):
        self.output = np.ones((weights_shape)) * self.constant_value
        return self.output

class UniformRandom:
    def __init__(self):
        self.output = None
    def initialize(self, weights_shape, fan_in, fan_out):
        self.output = np.random.uniform(0, 1, weights_shape)
        return self.output

class Xavier:
    def __init__(self):
        self.output = None
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        self.output = np.random.normal(0, sigma, weights_shape)
        return self.output

class He:
    def __init__(self):
        self.output = None
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        self.output = np.random.normal(0, sigma, weights_shape)
        return self.output