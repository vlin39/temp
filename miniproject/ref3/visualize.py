import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from beras.core import Tensor


def visualize_predictions(model: 'SequentialModel', train_inputs: Tensor, train_labels_ohe: np.ndarray, num_searching: int = 500) -> None:
    """
    param model: a neural network model (i.e. SequentialModel)
    param train_inputs: sample training inputs for the model to predict
    param train_labels_ohe: one-hot encoded training labels corresponding to train_inputs

    Displays 10 sample outputs the model correctly classifies and 10 sample outputs the model
    incorrectly classifies
    """

    rand_idx = np.random.choice(len(train_inputs), num_searching)
    rand_batch = train_inputs[rand_idx]
    probs = model(rand_batch)

    pred_classes = np.argmax(probs, axis=1)
    true_classes = np.argmax(train_labels_ohe[rand_idx], axis=1)

    right_idx = np.where(pred_classes == true_classes)
    wrong_idx = np.where(pred_classes != true_classes)

    right = np.reshape(rand_batch[right_idx], (-1, 28, 28))
    wrong = np.reshape(rand_batch[wrong_idx], (-1, 28, 28))

    right_pred_labels = true_classes[right_idx]
    wrong_pred_labels = pred_classes[wrong_idx]
    
    # Get confidence scores (max probabilities)
    right_confidences = np.max(probs[right_idx], axis=1)
    wrong_confidences = np.max(probs[wrong_idx], axis=1)

    assert len(right) >= 10, f"Found less than 10 correct predictions!"
    assert len(wrong) >= 10, f"Found less than 10 correct predictions!"

    fig, axs = plt.subplots(2, 10, figsize=(15, 8))
    fig.suptitle("Your Model's Predictions\n(PL = Predicted Label, Conf = Confidence)", fontsize=16)

    subsets = [right, wrong]
    pred_labs = [right_pred_labels, wrong_pred_labels]
    confidences = [right_confidences, wrong_confidences]
    row_labels = ["Correct Predictions", "Wrong Predictions"]
    colors = ['green', 'red']

    for r in range(2):
        for c in range(10):
            axs[r, c].imshow(subsets[r][c], cmap="Greys")
            axs[r, c].set_title(f"PL: {pred_labs[r][c]}\nConf: {confidences[r][c]:.2f}", 
                               color=colors[r], fontsize=10)
            plt.setp(axs[r, c].get_xticklabels(), visible=False)
            plt.setp(axs[r, c].get_yticklabels(), visible=False)
            axs[r, c].tick_params(axis="both", which="both", length=0)
        
        # Add row label
        axs[r, 0].text(-0.15, 0.5, row_labels[r], transform=axs[r, 0].transAxes, 
                       rotation=90, ha='center', va='center', fontsize=12, fontweight='bold',
                       color=colors[r])

    plt.tight_layout()
    plt.show()
