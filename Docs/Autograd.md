
# Autograd Engine Documentation

The **Autograd** engine provides reverse-mode automatic differentiation. It dynamically builds a computation graph as operations are performed on tensors, enabling the automatic calculation of gradients for complex mathematical models, such as neural networks.

## Core Engine API

These are the primary functions used to interact with and manage the automatic differentiation engine, gradient buffers, and the backward pass.

### Backward Pass (`backward`)

#### Definition:
Triggers the reverse-mode auto-differentiation process. Starting from a root tensor (usually a scalar loss), it performs a topological sort of the computation graph and propagates gradients backward to all leaf tensors where `requires_grad` is true.

#### Usage:
```cpp
backward(loss);
````


---

### Accumulate Gradient (`accumulate_grad`)

#### Definition:

Safely adds incoming gradients from a source tensor into a target tensor's gradient buffer. It automatically handles broadcasting rules — if the forward pass involved broadcasting, this function automatically reduces (sums) the gradients along the broadcasted dimensions to match the target's original shape.

#### Usage:

```cpp
accumulate_grad(target, incoming_grad);
```


---

### Ensure Gradient Buffer (`ensure_grad_buffer`)

#### Definition:

Allocates the underlying memory for a tensor's gradient buffer if it does not already exist. Optionally, it can fill the existing or newly created buffer entirely with zeros.

#### Usage:

* **Create buffer if missing, but do not overwrite existing data**:

  ```cpp
  ensure_grad_buffer(tensor, false);
  ```

* **Create buffer and strictly zero it out**:

  ```cpp
  ensure_grad_buffer(tensor, true);
  ```

#### Returns:

* `void`

---

### Tensor From Grad (`tensor_from_grad`)

#### Definition:

Extracts the raw memory of a tensor's gradient buffer and wraps it into a standard, usable `Tensor` object. This is primarily used internally during the backward pass to manipulate gradients mathematically.

#### Usage:

```cpp
Tensor grad_tensor = tensor_from_grad(self);
```

#### Returns:

* `Tensor`

---

## Computation Graph Nodes

When mathematical operations are performed on tensors that require gradients, the Autograd engine records these operations by instantiating specific **Node** objects.

### Gradient Function Base (`GradFn`)

#### Definition:

The abstract base structure representing a node in the dynamic computation graph. It stores references to the parent tensors (dependencies) and mandates an overridden `backward()` method to define the specific chain-rule derivative for an operation.

#### Usage:

```cpp
// Internal usage pattern
struct MyCustomGrad : GradFn {
    void backward(const Tensor& self) override {
        // Compute chain rule
    }
};
```


---

### Binary Operation Nodes

#### Definition:

Subclasses of `GradFn` that track operations between two tensors. They store references to Tensor `a` and Tensor `b` and calculate derivatives for both during the backward pass.

#### Examples:

* **GradAdd**: Distributes gradients equally to `a` and `b`.
* **GradMul**: Applies the product rule (`grad * b` for `a`, `grad * a` for `b`).
* **GradMatMul**: Applies matrix multiplication derivatives using transposes.

#### Usage:

```cpp
// Created automatically by the engine when calling operations like:
Tensor c = add(a, b);
```

---

### Unary Operation Nodes

#### Definition:

Subclasses of `GradFn` that track mathematical transformations applied to a single tensor. They apply standard calculus derivatives.

#### Examples:

* **GradExp**: Multiplies the incoming gradient by `exp(t)`.
* **GradLn**: Multiplies the incoming gradient by `1/t`.
* **GradRelu**: Passes gradients through only where the original tensor was `> 0`.

#### Usage:

```cpp
// Created automatically by the engine when calling operations like:
Tensor y = relu(x);
```

---

### Shape & Reduction Nodes

#### Definition:

Nodes that handle gradients for operations that change the shape or reduce the dimensions of a tensor, safely mapping the gradients back to the original physical layout.

#### Examples:

* **GradSum / GradMean**: Broadcasts the scalar/reduced gradient back to the original tensor shape.
* **GradPermute**: Reverses the memory permutation logic to route gradients to the correct original indices.
* **GradGather**: Uses scatter-add logic to route gradients back to specific indices of an embedding matrix.

#### Usage:

```cpp
// Created automatically by the engine when calling operations like:
Tensor total = sum(matrix, 0);
```

