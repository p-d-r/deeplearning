import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.loss = None
        self.subgradient = None
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        return np.sum(np.square(weights)) * self.alpha

class L1_Regularizer:
    def __init__(self, alpha):
        self.loss = None
        self.subgradient = None
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return np.sign(weights) * self.alpha

    def norm(self, weights):
        return np.sum(np.abs(weights)) * self.alpha

