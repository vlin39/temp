import numpy as np
from typing import List

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> List[Tensor]:
        return []

    def get_weight_gradients(self) -> List[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_total = np.mean((y_true - y_pred) ** 2, axis=-1)
        return Tensor(np.mean(mse_total, axis=0))

    def get_input_gradients(self) -> List[Tensor]:
        y_pred, y_true = self.inputs
        grad = 2 * (y_true - y_pred) / np.prod(y_pred.shape)
        return [-grad, np.zeros_like(grad)]


class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Categorical cross entropy forward pass!"""
        prod = y_true * np.log(np.clip(y_pred, self.epsilon, 1 - self.epsilon))
        prod_total = -np.sum(prod, axis=-1)
        return np.mean(prod_total, axis=0)

    def get_input_gradients(self) -> List[Tensor]:
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        n = y_pred.shape[0]
        grad = - (y_true / np.clip(y_pred, self.epsilon, 1 - self.epsilon)) / n
        return [grad, np.zeros_like(grad)]