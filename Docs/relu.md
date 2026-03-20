
# Activation Functions (ReLU Variants)

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. This module provides standard ReLU and its advanced variants (Leaky ReLU, Parametric ReLU) as both stateful layers and stateless functional operations.

## Standard ReLU (`Relu`)

### Definition

A stateful layer class (inheriting from `Module`) that applies the standard Rectified Linear Unit function. It outputs the input directly if it is positive, and outputs zero if it is negative. It has no learnable parameters.

### Usage

```cpp
// Initialize the layer
Relu activation_layer;

// Apply forward pass
Tensor output = activation_layer(input_tensor);
````

**Returns:**
`Tensor` (Same shape as input)

---

## Leaky ReLU (`LeakyRelu`)

### Definition

A stateless functional operation. Unlike standard ReLU, Leaky ReLU allows a small, non-zero, fixed gradient when the unit is not active (input < 0). This helps mitigate the "dying ReLU" problem where neurons get stuck outputting zero.
The `negative_slope` is fixed and not learned during training.

### Usage

```cpp
// Apply functionally with default slope (0.01)
Tensor output1 = LeakyRelu(input_tensor);

// Apply with a custom fixed slope
Tensor output2 = LeakyRelu(input_tensor, 0.05);
```

**Returns:**
`Tensor` (Same shape as input)

---

## Parametric ReLU (`PRelu`)

### Definition

A stateful layer class (inheriting from `Module`) that applies a Leaky ReLU where the negative slope is treated as a learnable parameter (often called `alpha` or `weight`). The slope is updated dynamically by the optimizer during the backward pass.
The parameter can be:

* **Shared:** A single slope value for all channels (`num_parameters = 1`).
* **Per-Channel:** A distinct slope learned independently for each channel (`num_parameters = C`).

### Usage

```cpp
// Initialize a PReLU layer with a single shared slope (default start = 0.25)
PRelu prelu_shared(1, 0.25);

// Initialize a PReLU layer with per-channel slopes (e.g., 64 channels)
PRelu prelu_channel(64, 0.25);

// Apply forward pass
Tensor output = prelu_shared(input_tensor);

// Extract learnable weights for the Optimizer
std::vector<Tensor*> params = prelu_shared.parameters();
```

**Returns:**
`Tensor` (Same shape as input)


