from collections import defaultdict
import numpy as np
from typing import List

from beras.core import Tensor

class BasicOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        for i in range(len(trainable_params)):
            if not trainable_params[i].trainable: continue
            trainable_params[i].assign(trainable_params[i]- grads[i] * self.learning_rate)


class RMSProp:
    def __init__(self, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-6) -> None:
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        ## TODO: Implement RMSProp optimization
        # pass
        for i in range(len(trainable_params)):
            if not trainable_params[i].trainable: continue
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grads[i] ** 2
            trainable_params[i].assign(trainable_params[i] - 
                grads[i] * self.learning_rate / (np.sqrt(self.v[i]) + self.epsilon)
            )


class Adam:
    def __init__(
        self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False
    ) -> None:

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        ## TODO: Implement Adam optimization
        ## HINT: Lab 2?
        # pass
        self.t += 1
        for i in range(len(trainable_params)):
            if not trainable_params[i].trainable: continue
            self.m[i] = self.beta_1 * self.m[i] + (1.0 - self.beta_1) * grads[i]
            self.v[i] = self.beta_2 * self.v[i] + (1.0 - self.beta_2) * grads[i] ** 2
            m_hat = self.m[i] / (1.0 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta_2 ** self.t)
            trainable_params[i] -=(
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )
