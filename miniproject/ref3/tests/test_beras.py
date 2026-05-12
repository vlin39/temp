#!/usr/bin/env python3
try:
    from .test_utils import setup_module_path, handle_cli_args
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_utils import setup_module_path, handle_cli_args
setup_module_path()
import numpy as np
import tensorflow as tf
import beras
import random
from sklearn.metrics import mean_squared_error
from beras.layers import Dense

"""
TESTING FILE FOR BERAS.PY

The tests in this file are for your layers, losses, activations, and metrics! 
However, these are the minimal set of tests to ensure your code is correct.

In fact, you are responsible for implementing the basic tests for sigmoid,
softmax, and CCE loss! You should add more tests to ensure your code is correct.

Note: All test functions should start with 'test_' to be automatically discovered
by our testing framework.
"""


def test_dense_forward():
    """Test the forward pass of your Dense layer"""

    # Instantiate your Dense layer and Keras Dense layer
    your_dense = Dense(10, 5)
    keras_dense = tf.keras.layers.Dense(units=5, input_shape=(10,))

    # Generate a random test input
    test_input = np.random.randn(10, 10)

    # Compute the forward pass of your Dense layer and Keras Dense layer
    your_out = your_dense(test_input)
    keras_out = keras_dense(test_input)
    
    # Compare the shapes of the output
    assert keras_out.shape == your_out.shape, f"Shapes differ: {keras_out.shape} vs {your_out.shape}"
    print("Forward pass successful")

def test_leaky_relu():
    """
    Test the forward pass of the LeakyReLU activation function.
    """

    # Instantiate your LeakyReLU activation function and Keras LeakyReLU activation function
    student_leaky_relu = beras.activations.LeakyReLU()
    leaky_relu = tf.keras.layers.LeakyReLU()

    # Generate a random test array
    test_arr = np.array(np.arange(-8,8),np.float64)

    # Compare the forward pass of your LeakyReLU activation function and Keras LeakyReLU activation function
    assert(all(np.isclose(student_leaky_relu(test_arr),leaky_relu(test_arr))))
    print("Leaky ReLU test passed!")


def test_sigmoid():
    """Use the Leaky ReLU test as a guide to test your Sigmoid activation!"""

    # TODO: Implement this test
    student_sigmoid = beras.activations.Sigmoid()
    sigmoid = tf.keras.activations.sigmoid
    test_arr = np.array(np.arange(-8,8),np.float64)
    assert(all(np.isclose(student_sigmoid(test_arr),sigmoid(test_arr))))
    print("Sigmoid test passed!")

def test_softmax():
    """Use the Leaky ReLU test as a guide to test your Softmax activation!"""

    # TODO: Implement this test
    student_softmax = beras.activations.Softmax()
    softmax = tf.keras.activations.softmax
    # test_arr = np.array(np.arange(-8,8),np.float64)
    # Reshape to 2D for TensorFlow compatibility (batch_size=1, features=16)
    # tf_test_arr = tf.constant(test_arr.reshape(1, -1))
    test_arr = np.random.rand(3, 4)
    test_arr = tf.constant(test_arr)
    # tf_result = softmax(tf_test_arr).numpy().flatten()  # Flatten back to 1D
    assert(np.all(np.isclose(student_softmax(test_arr), softmax(test_arr))))
    print("Softmax test passed!")

def test_mse_forward():
    """
    Test the forward pass of the MeanSquaredError loss function. 

    NOTE: We use sklearn's mean_squared_error to compute the solution MSE instead of tensorflow's MSE since it's more stable.
    """

    # Instantiate your MeanSquaredError loss function
    beras_mse = beras.MeanSquaredError()

    # Generate random test cases
    x = np.random.randint(0, 10, size=(2, 3))
    y = np.random.randint(0, 10, size=(2, 3))

    # Compute the solution MSE using sklearn (again because tensorflow's MSE is not stable)
    solution_mse = mean_squared_error(x, y)

    # Compare the solution MSE and your MSE
    assert np.allclose(solution_mse, beras_mse(x, y))

    print("MSE test passed!")

def test_cce():
    """Test the forward pass of the CategoricalCrossentropy loss function.
    
    TODO: We give you a lot of the code here since testing the CCE loss function is difficult.
    However, you still need to fill in the missing parts!
    """
    # Generate random test cases
    batch_size, num_classes = 5, 4

    true_labels = np.random.randint(0, num_classes, batch_size)
    y_true = np.eye(num_classes)[true_labels]
    
    # Random predictions (softmax normalized)
    logits = np.random.randn(batch_size, num_classes)
    y_pred = beras.activations.Softmax()(logits)
    
    # Our implementation
    our_loss = beras.losses.CategoricalCrossEntropy()(y_pred, y_true)
    
    # TensorFlow implementation (this uses slightly different syntax than your implementation)
    tf_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred).numpy().mean()
    
    # Should be very close
    np.testing.assert_almost_equal(our_loss, tf_loss, decimal=5)
    print("CCE test passed")

def test_categorical_accuracy():
    y_true = [[0, 0, 1], [0, 1, 0]]
    y_pred = np.random.uniform(0, 1, size=(2, 3))
    student_acc = beras.metrics.CategoricalAccuracy()(y_pred,y_true)
    acc = tf.keras.metrics.categorical_accuracy(y_true,y_pred)
    assert(student_acc == np.mean(acc))
    print("Categorical accuracy test passed")

# ============================================================================
# TODO: Add more tests to ensure your code is correct!
# ============================================================================


# ============================================================================
# TEST RUNNERS! DO NOT EDIT THIS SECTION!
# ============================================================================
try:
    from .test_utils import run_single_test, list_available_tests, run_all_tests
except ImportError:
    from tests.test_utils import run_single_test, list_available_tests, run_all_tests

# Convenience functions that wrap the utilities with this module's context
def run_test(test_name: str):
    """Run a single test by name. Usage: run_test('test_dense_forward')"""
    return run_single_test(test_name, globals(), 'test_beras')

def list_tests():
    """List all available tests in this module."""
    return list_available_tests(globals(), 'test_beras')

def run_all():
    """Run all tests in this module."""
    return run_all_tests(globals(), 'test_beras')

if __name__ == "__main__":
    handle_cli_args(globals(), 'test_beras')