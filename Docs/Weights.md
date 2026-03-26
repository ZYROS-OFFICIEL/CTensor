# Weight Initialization (`weights_init`)

The weight initialization module provides functions to set the starting values of a neural network's parameters before training begins. Proper initialization is critical for preventing vanishing or exploding gradients and ensuring fast, stable convergence.

All functions in this module modify the provided tensors **in-place**. Following standard conventions, function names ending with an underscore (e.g., `uniform_`) indicate an in-place operation.

---

## Basic Initialization

These functions fill a tensor with deterministic or randomly distributed values.

### Constant Fills (`zeros_`, `ones_`, `constant_`)

**Definition**  
Fills the target tensor(s) entirely with zeros, ones, or a specific constant value.

**Usage**
```cpp
Tensor t = Tensor::empty({3, 3});

zeros_(t);           // Fills with 0.0
ones_(t);            // Fills with 1.0
constant_(t, 3.14);  // Fills with 3.14

// Bulk application to all parameters in a model
std::vector<Tensor*> params = model.parameters();
zeros_(params);
````

**Returns**
`void` (Modifies memory in-place)

---

### Random Fills (`uniform_`, `normal_`)

**Definition**
Fills the tensor with values drawn from a standard probability distribution.
Uses a fast, thread-local Mersenne Twister (`std::mt19937`) for safe and efficient parallel execution.

**Usage**

```cpp
// Uniform distribution between [a, b]
uniform_(tensor, -1.0, 1.0);

// Normal (Gaussian) distribution
normal_(tensor, 0.0, 1.0); // mean = 0.0, std = 1.0
```

**Returns**
`void` (Modifies memory in-place)

---

## Advanced Network Initialization

These methods scale the initial weights based on tensor shape (using `fan_in` and `fan_out` of layers like Linear and Convolutional).

> **Note:**
> When applied to a bulk `std::vector<Tensor*>` (e.g., `model.parameters()`):

* Weight tensors (2D or more) receive advanced initialization
* Bias tensors (1D) are automatically initialized with `zeros_`

---

### Xavier / Glorot Initialization (`xavier_uniform_`, `xavier_normal_`)

**Definition**
Scales weights based on both input and output dimensions (`fan_in` and `fan_out`).

**Best for**
Networks using symmetric, non-linear activations like **Sigmoid** or **Tanh**.

**Usage**

```cpp
// Single tensor (default gain = 1.0)
xavier_uniform_(weight_tensor, 1.0);
xavier_normal_(weight_tensor, 1.0);

// Entire model
std::vector<Tensor*> params = model.parameters();
xavier_uniform_(params);
```

**Returns**
`void` (Modifies memory in-place)

---

### Kaiming / He Initialization (`kaiming_uniform_`, `kaiming_normal_`, `kaiming_init`)

**Definition**
Scales weights based on input dimension (`fan_in`), optimized for rectifier activations.

**Best for**

* ReLU
* Leaky ReLU
* PReLU

The optional parameter `a` represents the negative slope of the activation (e.g., `0.0` for ReLU).

**Usage**

```cpp
// Single tensor for ReLU (a = 0.0)
kaiming_uniform_(weight_tensor, 0.0);
kaiming_normal_(weight_tensor, 0.0);

// Entire model
std::vector<Tensor*> params = model.parameters();
kaiming_uniform_(params);

// Legacy wrapper
kaiming_init(params);
```

**Returns**
`void` (Modifies memory in-place)

---
