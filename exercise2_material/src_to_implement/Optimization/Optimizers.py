import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.updated_weight = None
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return self.updated_weight

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        self.updated_weight = weight_tensor + self.v
        return self.updated_weight

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.g = 0
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k = self.k + 1
        self.g = gradient_tensor
        self.v = self.mu * self.v + (1 - self.mu) * self.g
        self.r = self.rho * self.r + (1 - self.rho) * self.g ** 2
        bias_r = self.r / (1 - self.rho ** self.k)
        bias_v = self.v / (1 - self.mu ** self.k)
        self.updated_weights = weight_tensor - self.learning_rate * (bias_v / (np.sqrt(bias_r) + np.finfo(float).eps))
        return self.updated_weights