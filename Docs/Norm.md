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
