import numpy as np

from typing import Literal, List, Tuple
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size: int, output_size: int, initializer: DENSE_INITIALIZERS = "normal") -> None:
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> List[Tensor]:
        return [self.w, self.b]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        return Tensor(x @ self.w + self.b)

    def get_input_gradients(self) -> List[Tensor]:
        return [self.w]

    def get_weight_gradients(self) -> List[Tensor]:
        x = self.inputs[0]
        ## note that the actual weight gradient has shape [batch, out, in, out]
        ## grad = x[:, None, :, None] * np.eye(w.shape[1])[None, :, None, ;]
        ## bc y has shape [batch, out] and we differentiate by w [in, out]
        ## this is just a nice shorthand because sum(grad, axis=1)=wgrads!
        ## the sum in the composition matmul is condensed into the sum above
        ## and we're just left with element-wise
        wgrads = np.ones_like(self.w) * np.expand_dims(x, axis=-1)
        # Also a valid solution here
        # wgrads = np.repeat(np.expand_dims(x, axis=-1), self.w.shape[1], axis=-1)
        bgrads = np.ones_like(self.b)
        return [Tensor(wgrads), Tensor(bgrads)]

    @staticmethod
    def _initialize_weight(initializer: DENSE_INITIALIZERS, input_size: int, output_size: int) -> Tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        io_size = (input_size, output_size)
        w_init = np.zeros(io_size)
        b_init = np.zeros((1, output_size))

        if initializer == "normal":
            w_init = np.random.normal(size=io_size)
        if initializer == "xavier":
            std = np.sqrt(2 / sum(io_size))
            w_init = np.random.normal(size=io_size) * std
        if initializer == "kaiming":
            std = np.sqrt(2 / input_size)
            w_init = np.random.normal(size=io_size) * std
        if initializer == "xavier uniform":
            std = np.sqrt(6 / sum(io_size))
            w_init = np.random.uniform(-std, std, size=io_size)
        if initializer == "kaiming uniform":
            std = np.sqrt(6 / input_size)
            w_init = np.random.uniform(-std, std, size=io_size)

        return Variable(w_init), Variable(b_init)
