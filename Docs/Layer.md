# Layers

Layers are the fundamental, stateful building blocks of neural networks. They inherit from the base `Module` class and typically contain learnable parameters (weights and biases) that are updated during the training process, alongside a `forward` method that defines their computation.

---

## Linear / Fully Connected Layer (`Linear`)

### Definition

Applies a linear transformation to the incoming data:

```

y = xW^T + b

````

This is commonly known as a dense or fully connected layer. It maintains:
- A learnable weight matrix of shape `[out_features, in_features]`
- An optional learnable bias vector of shape `[out_features]`

### Usage

```cpp
// Initialize a linear layer: 128 input features, 64 output features, with bias
Linear fc_layer(128, 64, true, DType::Float32);

// Forward pass (Input shape: [Batch, 128])
Tensor output = fc_layer(input_tensor);

// Retrieve parameters for the optimizer
std::vector<Tensor*> params = fc_layer.parameters();
````

### Returns

* `Tensor` with shape: `[Batch, out_features]`

---

## Flatten Layer (`Flatten`)

### Definition

A utility layer that flattens a contiguous range of dimensions into a single dimension. It contains no learnable parameters.

This is most frequently used to transition from spatial data (like the output of convolutional layers) to flat data (expected by linear layers).

By default, it flattens from dimension `1` (skipping the batch dimension at `0`) to the last dimension `-1`.

For example:

```
[Batch, Channels, Height, Width] → [Batch, Channels * Height * Width]
```

### Usage

```cpp
// Initialize default flatten layer (flattens dim 1 to -1)
Flatten flatten_layer;

// Initialize custom flatten layer (e.g., flatten only dims 2 and 3)
Flatten custom_flatten(2, 3);

// Forward pass
Tensor output = flatten_layer(input_tensor);
```

### Returns

* `Tensor` (reshaped tensor with flattened dimensions)

