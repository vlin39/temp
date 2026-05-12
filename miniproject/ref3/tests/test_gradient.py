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
import beras.gradient_tape as gradient_tape
import beras.layers as layers
import beras.losses as losses
import beras.core as core

"""
TESTING FILE FOR GRADIENT.PY

The tests in this file are for your gradient computation and backpropagation! 
However, these are the minimal set of tests to ensure your code is correct.

Note: All test functions should start with 'test_' to be automatically discovered
by our testing framework.
"""

def test_multi_layer_gradients():
    """
    Test gradient computation through multiple layers.
    """    
    # Create a two-layer network
    layer1 = layers.Dense(3, 2, initializer="normal")
    layer2 = layers.Dense(2, 1, initializer="normal")
    
    # Set random seed for reproducible weights
    np.random.seed(42)
    
    layer2.w.assign(np.random.normal(0, 0.1, (2, 1)))
    
    # Handle both 1D and 2D bias implementations for layer2
    if len(layer2.b.shape) == 1:
        layer2.b.assign(np.zeros(1))    
    else:
        layer2.b.assign(np.zeros((1, 1)))
    
    # Fixed input and target
    x = core.Tensor([[1.0, 0.5, -0.3]])  # 1x3 input
    y_true = core.Tensor([[1.5]])         # 1x1 target
    
    loss_fn = losses.MeanSquaredError()
    
    # Forward pass with gradient tape
    with gradient_tape.GradientTape() as tape:
        hidden = layer1(x)
        output = layer2(hidden)
        loss = loss_fn(output, y_true)  # MSE takes (y_pred, y_true)
    
    # Get all trainable variables
    all_variables = layer1.trainable_variables + layer2.trainable_variables
    
    # Compute gradients
    gradients = tape.gradient(loss, all_variables)
    
    # Check that we got gradients for all variables
    assert len(gradients) == 4, f"Expected 4 gradients (2 layers x 2 variables each), got {len(gradients)}"
    
    # Check that no gradients are None
    for i, grad in enumerate(gradients):
        assert grad is not None, f"Gradient {i} should not be None"
    
    # Check gradient shapes match variable shapes
    for i, (grad, var) in enumerate(zip(gradients, all_variables)):
        assert grad.shape == var.shape, f"Gradient {i} shape {grad.shape} doesn't match variable shape {var.shape}"
    
    print("Multi-layer gradient computation test passed!")


def test_deterministic_gradients():
    """
    Test with completely deterministic setup and manually computed expected gradients.
    """

    # Create simple network: one dense layer (1->1) with known weights
    layer = layers.Dense(1, 1, initializer="zero")
    layer.w.assign([[2.0]])     # weight = 2
    
    # Handle both 1D and 2D bias implementations
    if len(layer.b.shape) == 1:
        layer.b.assign([1.0])   # 1D bias implementation
    else:
        layer.b.assign([[1.0]]) # 2D bias implementation
    
    # Simple input and target
    x = core.Tensor([[3.0]])       # input = 3
    y_true = core.Tensor([[10.0]]) # target = 10
    
    loss_fn = losses.MeanSquaredError()
    
    with gradient_tape.GradientTape() as tape:
        # Forward pass: y_pred = x * w + b = 3 * 2 + 1 = 7
        y_pred = layer(x)
        # Loss using MSE: Loss = (y_true - y_pred)^2 = (10 - 7)^2 = 9
        loss = loss_fn(y_pred, y_true)
    
    gradients = tape.gradient(loss, layer.trainable_variables)
    
    # BREAKING DOWN THE MANUAL GRADIENT COMPUTATION FOR MSE LOSS:
    # MSE loss: L = (y_true - y_pred)^2
    # You should calculate the gradient manually and will get -6 as the gradient.
    
    # dy_pred/dw = x = 3 (input)
    # dL/dw = dL/dy_pred * dy_pred/dw = -6 * 3 = -18
    # dy_pred/db = 1
    # dL/db = dL/dy_pred * dy_pred/db = -6 * 1 = -6
    
    expected_w_grad = -18.0
    expected_b_grad = -6.0
    
    # Extract scalar from gradient
    w_grad = gradients[0][0, 0]  
    
    # Handle both 1D and 2D bias gradient shapes
    if len(gradients[1].shape) == 1:
        b_grad = gradients[1][0]    # 1D bias gradient
    else:
        b_grad = gradients[1][0, 0] # 2D bias gradient  
    
    assert abs(w_grad - expected_w_grad) < 1e-6, f"Weight gradient {w_grad} doesn't match expected {expected_w_grad}"
    assert abs(b_grad - expected_b_grad) < 1e-6, f"Bias gradient {b_grad} doesn't match expected {expected_b_grad}"
    
    print("Deterministic gradient computation test passed!")

# TODO: Add more tests for gradient computation as you see fit!

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
    """Run a single test by name. Usage: run_test('test_multi_layer_gradients')"""
    return run_single_test(test_name, globals(), 'test_gradient')

def list_tests():
    """List all available tests in this module."""
    return list_available_tests(globals(), 'test_gradient')

def run_all():
    """Run all tests in this module."""
    return run_all_tests(globals(), 'test_gradient')

if __name__ == "__main__":
    handle_cli_args(globals(), 'test_gradient')