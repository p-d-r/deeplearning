import numpy as np
import sys
sys.path.append("..")
from src_to_implement.Layers import SoftMax


class CrossEntropyLoss:

    def __init__(self):
        self.hei = None


    def forward(self, prediction_tensor, label_tensor):
        log_likelihood = np.log(prediction_tensor + np.finfo(float).eps) * label_tensor
        self.prediction_tensor = prediction_tensor
        return - np.sum(log_likelihood)

    def backward(self, label_tensor):
        return - label_tensor / (self.prediction_tensor + np.finfo(float).eps)