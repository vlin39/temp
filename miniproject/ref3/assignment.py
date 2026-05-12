from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np
from visualize import visualize_predictions

from beras.model import SequentialModel

def get_model() -> SequentialModel:
    model = SequentialModel(
        [
            Dense(784, 64, initializer="kaiming"),
            LeakyReLU(),
            Dense(64, 10, initializer="kaiming"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    return Adam(0.01)

def get_loss_fn():
    return MeanSquaredError()

def get_acc_fn():
    return CategoricalAccuracy()


if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel
    model = SequentialModel(
        [
            Dense(784, 64, initializer="kaiming"),
            LeakyReLU(),
            Dense(64, 10, initializer="kaiming"),
            Softmax(),
        ]
    )

    # 2. Compile the model
    model.compile(
        optimizer=get_optimizer(),
        loss_fn=get_loss_fn(),
        acc_fn=get_acc_fn(),
    )
    
    # 3. Load and preprocess the data
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()
    ohe = OneHotEncoder()
    concat_labels = np.concatenate([train_labels, test_labels], axis=-1)
    ohe.fit(concat_labels)
    # 4. Train the model
    
    model.fit(train_inputs, ohe(train_labels), epochs=1, batch_size=512)

    # 5. Evaluate the model
    metrics, predictions = model.evaluate(test_inputs, ohe(test_labels), batch_size=512)
    
    # 6. save the predictions using np.save
    np.save('predictions_.npy', predictions)

    # 7. Call visualize_predictions
    visualize_predictions(model, train_inputs, ohe(train_labels))
        
