# Tensor Operations API

This document outlines the mathematical operations and neural network functions available for `Tensor` objects. Standard C++ operators are overloaded for convenience, and all operations automatically integrate with the Autograd engine.

## Core Operations

### Binary Operations
These functions perform element-wise calculations or matrix multiplication between two tensors. Tensors must be located on the same device.

* `add(a, b)`: Element-wise addition.
* `sub(a, b)`: Element-wise subtraction.
* `mul(a, b)`: Element-wise multiplication.
* `div(a, b)`: Element-wise division.
* `pow(a, b)`: Element-wise power.
* `matmul(a, b)`: Matrix multiplication.

### Scalar Operations
These functions apply a mathematical operation between a tensor and a standard double-precision scalar.

* `add_scalar(a, scalar)`
* `sub_scalar(a, scalar)`
* `sub_scalar_rev(scalar, a)`: Computes `scalar - a`.
* `mul_scalar(a, scalar)`
* `div_scalar(a, scalar)`
* `div_scalar_rev(scalar, a)`: Computes `scalar / a`.
* `pow_scalar(a, scalar)`
* `pow_scalar_rev(scalar, a)`: Computes `scalar ^ a`.

### Unary Operations and Activations
These functions apply element-wise mathematical transformations to a single tensor.

* **Basic Math**: `abs()`, `log()`, `ln()`, `exp()`, `sqrt()`.
* **Trigonometry**: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `sinh()`, `cosh()`, `tanh()`.
* **Activations**: `sigmoid()`, `relu()`, `softplus()`.

### Reductions
These functions reduce the dimensions of a tensor. By default (`dim = -1`), they reduce across the entire flattened tensor.

* `sum(a, dim)`
* `mean(a, dim)`
* `max(a, dim)`
* `min(a, dim)`
* `argmax(a, dim)`: Returns the indices of the maximum values as `Int32`.

### Logical Comparisons
Element-wise boolean comparisons. These return a tensor of boolean type. Overloads exist for both `(Tensor, Tensor)` and `(Tensor, double)`.

* `lt(a, b)`: Less than.
* `le(a, b)`: Less than or equal.
* `gt(a, b)`: Greater than.
* `ge(a, b)`: Greater than or equal.
* `eq(a, b)`: Equal to.
* `ne(a, b)`: Not equal to.

### Utility
* `cat(tensors, dim)`: Concatenates a vector of tensors along the specified dimension.

## C++ Operator Overloading

Standard C++ operators are overloaded to natively wrap the core mathematical and logical API. 

* **Arithmetic**: `+`, `-`, `*`, `/`, `^` (and their respective assignment operators `+=`, `-=`, etc.).
* **Unary**: `-A` maps to `mul_scalar(A, -1.0)`.
* **Comparisons**: `<`, `<=`, `>`, `>=`, `==`, `!=`.

## Hardware Execution and Autograd

Hardware dispatch is resolved automatically.

If any input tensor to an operation has `requires_grad` enabled, the resulting output tensor will also require gradients. The operation is then recorded in the computational graph with the corresponding backward function (e.g., `GradAdd`, `GradRelu`) for automatic differentiation.