# Norms & Normalization (norm)

The `norm` module provides mathematical utilities for calculating tensor norms (L1, L2, general Lp-norms) and normalizing tensors based on those calculated distances. These operations are fully differentiable and build upon the core autograd engine.

---

## Tensor Norm (`norm`)

### Definition

Computes the p-norm of a tensor along a specified dimension.

- **p**: The exponent value in the norm formulation  
  - `1.0` → L1 (Manhattan norm)  
  - `2.0` → L2 (Euclidean norm)
- **dim**: The dimension to reduce  
  - If `dim = -1` and the tensor is multi-dimensional, the tensor is flattened and the global norm is computed across all elements.
- **keepdim**: A boolean flag  
  - If `true`, the reduced dimension is retained with size `1`, enabling broadcasting with the original tensor.

### Usage

```cpp
// Compute the global L2 norm (Euclidean norm) of a tensor
Tensor global_l2 = norm(input_tensor, 2.0, -1, false);

// Compute the L1 norm along dimension 1, keeping the dimension for broadcasting
Tensor l1_dim1 = norm(input_tensor, 1.0, 1, true);
````

### Returns

* `Tensor`

---

## Lp Normalization (`Lp_Norm`)

### Definition

Scales the input tensor by dividing it by its Lp-norm along the specified dimension.

This is most commonly used for **L2 normalization**, where vectors are scaled to have a unit length of `1`.

A small **epsilon (`eps`)** value is automatically added to the denominator to prevent `NaN` or `Inf` errors caused by division by zero.

### Usage

```cpp
// Apply L2 normalization globally across the tensor
Tensor l2_normalized = Lp_Norm(input_tensor, 2.0, -1, 1e-12);

// Apply L2 normalization along a specific feature dimension (e.g., dim = 1)
Tensor feature_normalized = Lp_Norm(input_tensor, 2.0, 1, 1e-12);
```

### Returns

* `Tensor` (same shape as the input tensor)

---

## Layer Normalization (`LayerNorm`)

### Definition

A learnable normalization **layer** — unlike `norm` and `Lp_Norm` above, which are stateless functions.

`LayerNorm` standardizes each sample independently across its **last dimension**: it subtracts that vector's mean, divides by its standard deviation, then applies a learnable per-feature scale and shift.

```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

Because the statistics come from a single sample rather than the batch, LayerNorm behaves identically in training and inference — no running buffers, no mode switch, no dependence on batch size. That is what makes it the normalization of choice for transformers and recurrent models, where [BatchNorm](BatchNorm.md) is awkward or unusable.

### Constructor

```cpp
LayerNorm(int normalized_shape, double eps = 1e-5);
```

| Argument | Meaning |
|---|---|
| `normalized_shape` | Size of the last dimension — the model width |
| `eps` | Added to the variance before the square root, for numerical stability |

`weight` is initialized to ones and `bias` to zeros, both of shape `[normalized_shape]` and both `Float32` with gradients enabled, so the layer starts as a pure standardization and learns its scale from there.

### Usage

```cpp
// Normalize over a 768-wide feature dimension
LayerNorm ln(768, 1e-5);

Tensor out = ln(x);        // [B, S, 768] -> [B, S, 768]

// Parameters, for the optimizer
auto params = ln.parameters();                  // { &weight, &bias }
auto named  = ln.named_parameters("ln_1.");     // "ln_1.weight", "ln_1.bias"
```

### Returns

* `Tensor` (same shape as the input tensor)

> The forward pass is composed entirely of differentiable core ops (`mean`, `sqrt`, arithmetic), so autograd handles the backward pass without a dedicated node. A single-pass fused kernel also exists for inference — `layer_norm_avx2_f32` and friends, see [Backend](Backend.md).

---

## Choosing a Normalization

| | Normalizes over | Batch-dependent | Typical use |
|---|---|---|---|
| `LayerNorm` | Last dimension of each sample | No | Transformers, sequence models |
| [`BatchNorm`](BatchNorm.md) | Batch (and spatial) axes per channel | Yes | CNNs |
| `Lp_Norm` | Any dimension, no parameters | No | Feature/embedding scaling |
