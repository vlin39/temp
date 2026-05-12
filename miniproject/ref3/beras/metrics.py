import numpy as np

from beras.core import Callable, Tensor


class CategoricalAccuracy(Callable):
    def forward(self, probs: Tensor, labels: Tensor) -> float:
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        pred_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(labels, axis=1)
        return np.mean(pred_classes == true_classes)
