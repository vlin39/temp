from abc import abstractmethod
from collections import defaultdict
from typing import Union, Dict, List, Tuple, Any, Optional

from beras.core import Diffable, Tensor, Variable, Callable
from beras.gradient_tape import GradientTape
import numpy as np

def print_stats(stat_dict: Dict[str, float], batch_num: Optional[int] = None, num_batches: Optional[int] = None, epoch: Optional[int] = None, avg: bool = False) -> None:
    """
    Given a dictionary of names statistics and batch/epoch info,
    print them in an appealing manner. If avg, display stat averages.

    :param stat_dict: dictionary of metrics to display
    :param batch_num: current batch number
    :param num_batches: total number of batches
    :param epoch: current epoch number
    :param avg: whether to display averages
    """
    title_str = " - "
    if epoch is not None:
        title_str += f"Epoch {epoch+1:2}: "
    if batch_num is not None:
        title_str += f"Batch {batch_num+1:3}"
        if num_batches is not None:
            title_str += f"/{num_batches}"
    if avg:
        title_str += f"Average Stats"
    print(f"\r{title_str} : ", end="")
    op = np.mean if avg else lambda x: x
    print({k: np.round(op(v), 4) for k, v in stat_dict.items()}, end="")
    print("   ", end="" if not avg else "\n")


def update_metric_dict(super_dict: Dict[str, List[float]], sub_dict: Dict[str, float]) -> None:
    """
    Appends the average of the sub_dict metrics to the super_dict's metric list

    :param super_dict: dictionary of metrics to append to
    :param sub_dict: dictionary of metrics to average and append
    """
    for k, v in sub_dict.items():
        super_dict[k] += [np.mean(v)]


class Model(Diffable):

    def __init__(self, layers: List[Diffable]) -> None:
        """
        Initialize all trainable parameters and take layers as inputs
        """
        # Initialize all trainable parameters
        self.layers = layers

    @property
    def weights(self) -> List[Tensor]:
        """
        Return the weights of the model by iterating through the layers
        """
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

    def compile(self, optimizer: Diffable, loss_fn: Diffable, acc_fn: Callable) -> None:
        """
        "Compile" the model by taking in the optimizers, loss, and accuracy functions.
        In more optimized DL implementations, this will have more involved processes
        that make the components extremely efficient but very inflexible.
        """
        self.optimizer      = optimizer
        self.compiled_loss  = loss_fn
        self.compiled_acc   = acc_fn

    def fit(self, x: Tensor, y: Union[Tensor, np.ndarray], epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Trains the model by iterating over the input dataset and feeding input batches
        into the batch_step method with training. At the end, the metrics are returned.
        """
        agg_metrics = defaultdict(lambda: [])
        num_batches = x.shape[0] // batch_size
        for e in range(epochs):
            epoch_metrics = defaultdict(lambda: [])
            for b, b1 in enumerate(range(batch_size, x.shape[0] + 1, batch_size)):
                b0 = b1 - batch_size
                batch_metrics = self.batch_step(x[b0:b1], y[b0:b1], training=True)
                update_metric_dict(epoch_metrics, batch_metrics)
                print_stats(batch_metrics, b, num_batches, e)
            update_metric_dict(agg_metrics, epoch_metrics)
            print_stats(epoch_metrics, epoch=e, avg=True)
        return agg_metrics

    def evaluate(self, x: Tensor, y: Union[Tensor, np.ndarray], batch_size: int) -> Tuple[Dict[str, float], np.ndarray]:
        """
        X is the dataset inputs, Y is the dataset labels.
        Evaluates the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method. At the end, the metrics are returned. Should be called on
        the testing set to evaluate accuracy of the model using the metrics output from the fit method.

        NOTE: This method is almost identical to fit (think about how training and testing differ --
        the core logic should be the same)
        """
        ## TODO: Implement evaluate similarly to fit. Try to match the printing/aggregation logic. 
        agg_metrics = defaultdict(lambda: [])
        predictions = []
        batch_num = x.shape[0] // batch_size
        batch_num += 1 if x.shape[0] % batch_size != 0 else 0
        for b in range(batch_num):
            b0 = b*batch_size
            b1 = (b+1)*batch_size
            batch_metrics, preds = self.batch_step(x[b0:b1], y[b0:b1], training=False)
            update_metric_dict(agg_metrics, batch_metrics)
            print_stats(batch_metrics, b, batch_num)
            predictions.extend(np.argmax(preds, axis=1))

        agg_metrics = {k: np.mean(v) for k, v in agg_metrics.items()}
        print_stats(agg_metrics, avg=True)
        return agg_metrics, np.reshape(predictions, (-1,))

    def get_input_gradients(self) -> List[Tensor]:
        return super().get_input_gradients()

    def get_weight_gradients(self) -> List[Tensor]:
        return super().get_weight_gradients()
    
    @abstractmethod
    def batch_step(self, x: Tensor, y: Tensor, training: bool = True) -> Union[Dict[str, float], Tuple[Dict[str, float], Tensor]]:
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        raise NotImplementedError("batch_step method must be implemented in child class")

class SequentialModel(Model):
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass in sequential model. It's helpful to note that layers are initialized in beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        ## TODO: What does it mean to call the model?
       
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_step(self, x: Tensor, y: Tensor, training: bool = True) -> Union[Dict[str, float], Tuple[Dict[str, float], Tensor]]:
        """Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()"""
        ## TODO: Compute loss and accuracy for a batch. Return as a dictionary with keys 'loss' and 'acc'
        ## If training, then also update the gradients according to the optimizer
        with GradientTape() as tape:
            pred = self.forward(x)
            loss = self.compiled_loss(pred, y)
        if training:
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(self.trainable_variables, grads)
        acc = self.compiled_acc(pred, y)
        if training:
            return {"loss": loss, "acc": acc}
        else:
            return {"loss": loss, "acc": acc}, pred
