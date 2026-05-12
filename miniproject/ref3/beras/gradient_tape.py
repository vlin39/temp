from collections import defaultdict
from typing import List, Optional, Dict, Any

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self) -> None:
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: Dict[int, Optional[Diffable]] = defaultdict(lambda: None)

    def __enter__(self) -> 'GradientTape':
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: List[Tensor]) -> List[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        while queue:
            curr_output = queue.pop(0)
            curr_id = id(curr_output)
            if curr_id not in self.previous_layers.keys(): #if the current output was not produced by a layer operation recorded by the gradient tape
                continue
            curr_grad = grads[curr_id] #the upstream gradient amassed up to current output
            prev_layer = self.previous_layers[curr_id]
            for v, g in zip(prev_layer.inputs, prev_layer.compose_input_gradients(curr_grad)):
                grads[id(v)] = [g] #for the weird case where an input is trainable? e.g. input optimization lab
                queue += [v]
            for v, g in zip(prev_layer.weights, prev_layer.compose_weight_gradients(curr_grad)):
                grads[id(v)] = [g]
                queue += [v]

        ## TODO: Retrieve the sources and make sure that all of the sources have been reached
        out_grads = [grads[id(source)][0] for source in sources]
        disconnected = [f"var{i}" for i, grad in enumerate(out_grads) if grad is None]

        if disconnected:
            print(f"Warning: The following tensors are disconnected from the target graph: {disconnected}")

        return out_grads