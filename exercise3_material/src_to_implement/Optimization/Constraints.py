
class L2_Regularizer:
    def __init__(self, alpha):
        self.loss = None
        self.subgradient = None
        self.regularization_weight = alpha


    def calculate_gradient(self, weights):
        return self.subgradient

    def norm(self, weights):
        return self.loss

class L1_Regularizer:
    def __init__(self, alpha):
        self.loss = None
        self.subgradient = None
        self.regularization_weight = alpha

    def calculate_gradient(self, weights):
        return self.subgradient

    def norm(self, weights):
        return self.loss

