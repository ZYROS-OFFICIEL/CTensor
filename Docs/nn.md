
# Neural Network Module (torch::nn)

The `torch::nn` module provides a high-level API for building and training neural networks. It wraps the core tensor engine into familiar object-oriented modules, data loaders, and training utilities.

---

## Data Loading

The DataLoader provides an iterable interface over a dataset, automatically handling batching and shuffling.

### DataLoader Definition

Wraps a TensorDataset to yield batches of data and targets. It supports standard C++ range-based for-loops using an internal iterator.

### Usage

```cpp
// Initialize a dataloader with a dataset, batch size of 32, and shuffling enabled
torch::DataLoader loader(my_dataset, 32, true);

// Iterate through batches
for (auto& batch : loader) {
    Tensor x = batch.data;
    Tensor y = batch.target;
    // ... training logic ...
}
````

**Returns:**
Batch struct containing `Tensor data` and `Tensor target`

---

## Layers & Modules

These are the fundamental building blocks of neural networks, aliased into the `torch::nn` namespace for standard access.

### Core Layers

**Definition:**
Standard stateful and stateless network layers that inherit from the base `Module` class.

**Available Layers:**

* `nn::Linear`: A fully connected (dense) layer.
* `nn::Conv2d`: A 2D convolutional layer for spatial data (images).
* `nn::Flatten`: A utility layer to reshape multi-dimensional spatial tensors into 1D vectors for linear layers.
* `nn::Module`: The abstract base class for all custom neural network models.

---

## Functional API (nn::functional)

**Definition:**
Stateless functions, primarily used for applying activation functions to tensors during the forward pass.

### Usage

```cpp
Tensor out1 = torch::nn::functional::relu(x);
Tensor out2 = torch::nn::functional::sigmoid(x);
```

**Returns:**
`Tensor`

---

## Loss Functions

### Cross Entropy Loss (CrossEntropyLoss)

**Definition:**
Calculates the cross-entropy loss between predictions and target labels. This implementation includes an automatic shape-correction mechanism that unsqueezes 1D target tensors `[B]` into 2D tensors `[B, 1]` to prevent dimension mismatch errors during loss calculation.

### Usage

```cpp
torch::nn::CrossEntropyLoss criterion;
Tensor loss = criterion(predictions, targets);
```

**Returns:**
`Tensor` (Scalar loss value)

---

## Training Utilities

A collection of helper functions to manage parameters, gradients, and weight initialization.

### Combine Parameters (combine_params)

**Definition:**
A variadic template utility that gathers all trainable parameters (weight and bias tensors) from multiple modules into a single flattened vector. This is essential for passing all model weights to an optimizer.

### Usage

```cpp
auto all_params = torch::nn::combine_params(layer1, layer2, layer3);
```

**Returns:**
`std::vector<Tensor*>`

---

### Gradient Clipping (utils::clip_grad_norm_)

**Definition:**
Prevents the "exploding gradient" problem by calculating the global L2 norm of all provided parameters' gradients. If the total norm exceeds the `max_norm` threshold, it scales all gradients down proportionally.

### Usage

```cpp
torch::nn::utils::clip_grad_norm_(model_params, 1.0);
```

**Returns:**
`void` (Modifies gradient buffers in-place)

---

### Weight Initialization (init::kaiming_uniform_)

**Definition:**
Applies Kaiming (He) uniform initialization to a list of parameters. This helps maintain variance across layers during the forward pass, particularly for networks utilizing ReLU activations.

### Usage

```cpp
torch::nn::init::kaiming_uniform_(model_params);
```

**Returns:**
`void` (Modifies tensor data in-place)
