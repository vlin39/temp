import numpy as np
from typing import List

from beras.core import Diffable, Tensor

class Activation(Diffable):
    @property
    def weights(self) -> List[Tensor]: 
        return []

    def get_weight_gradients(self) -> List[Tensor]: 
        return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Leaky ReLu forward propagation!"""
        out = np.copy(x)
        out[x < 0] *= self.alpha
        return Tensor(out)

    def get_input_gradients(self) -> List[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """

        x, y = self.inputs + self.outputs
        out = np.copy(x)  ## Hint

        out[out > 0] = 1
        out[out < 0] = self.alpha

        return [Tensor(out)]

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self) -> None:
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1/(1 + np.exp(-x)))

    def get_input_gradients(self) -> List[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        x, y = self.inputs + self.outputs
        return [y * (1 - y)]

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [1470]: Implement for default output activation to bind output to 0-1

    def forward(self, x: Tensor) -> Tensor:
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        shiftx = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shiftx)
        outs = exps / np.sum(exps, axis=-1, keepdims=True)
        return Tensor(outs)

    def get_input_gradients(self) -> List[Tensor]:
        """Softmax input gradients!"""
        # https://stackoverflow.com/questions/48633288/how-to-assign-elements-into-the-diagonal-of-a-3d-matrix-efficiently
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        for b in range(bn):
            out = y[b]
            grad[b] = -np.outer(out, out)
            np.fill_diagonal(grad[b], out * (1 - out))
        return [Tensor(grad)]

if __name__ == "__main__":
    leaky = LeakyReLU(alpha=.5)