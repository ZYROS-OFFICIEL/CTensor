
# Tensors

Tensors are mathematical tools used to represent **N-dimensional arrays**.  
They are essentially a generalization of **vectors** and **matrices** to **N dimensions**.

---

# Tensor Metadata

In **CTensor**, tensors contain metadata that defines:

- Their **physical memory layout**
- Their **data type**
- Their **autograd behavior**

Below are the primary metadata properties and how to access them via the `Tensor` object.

---

## Shape (`shape`)

**Definition**

The size of the tensor along each of its dimensions.

Example:  
A 2D matrix might have a shape of `[3, 4]`.

**Access**

```cpp
tensor.shape()
````

Returns:

```
std::vector<size_t>
```

You can also print it directly using:

```cpp
tensor.print_shape()
```

---

## Data Type (`dtype`)

**Definition**

Specifies the underlying numeric type of the tensor elements.

Examples:

* `Float32`
* `Int32`
* `Bool`

**Access**

```cpp
tensor._dtype()
```

Returns:

```
DType
```

To get the **byte size of a single element**:

```cpp
tensor.dtype_bytes()
```

---

## Total Elements (`numel`)

**Definition**

The total number of scalar elements contained in the tensor.

This is calculated as:

```
product of all dimensions in the shape
```

**Access**

```cpp
tensor.numel()
```

Returns:

```
size_t
```

---

## Device (`device`)

**Definition**

Specifies the hardware where the tensor's memory is allocated.

Examples:

* `CPU`
* `GPU`

**Access**

```cpp
tensor.device()
```

Returns:

```
Device
```

---

## Requires Gradient (`requires_grad`)

**Definition**

A boolean flag that determines whether the **Autograd engine** should track operations performed on this tensor in order to compute gradients during `.backward()`.

**Access**

```cpp
tensor.requires_grad()
```

Returns:

```
bool
```

Modify it in-place:

```cpp
tensor.requires_grad_(true)
tensor.requires_grad_(false)
```

---

## Contiguity (`is_contiguous`)

**Definition**

Indicates whether the tensor's data is stored **densely in memory using standard C-order (row-major)** without gaps.

Operations such as:

* **permuting**
* **slicing**

may produce **non-contiguous tensors**.

**Access**

```cpp
tensor.is_contiguous()
```

Returns:

```
bool
```

To force the tensor to become contiguous in memory:

```cpp
tensor.contiguous()
```

# Initialization & Factory Methods

These are static methods used to easily create new tensors from scratch.

---

## Zeros (`zeros`)

**Definition**

Creates a new tensor of the specified shape filled entirely with zeros.

**Usage**

```cpp
Tensor::zeros({3, 4}, DType::Float32)
```

Returns:

```
Tensor
```

---

## Ones (`ones`)

**Definition**

Creates a new tensor of the specified shape filled entirely with ones.

**Usage**

```cpp
Tensor::ones({2, 2}, DType::Float32)
```

Returns:

```
Tensor
```

---

## Random (`rand`)

**Definition**

Creates a new tensor filled with uniformly distributed random numbers between `0.0` and `1.0`.

**Usage**

```cpp
Tensor::rand({3, 3})
```

Returns:

```
Tensor
```

---

## Arange (`arange`)

**Definition**

Returns a 1-D tensor with values from the interval `[start, end)` separated by a common difference `step`.

**Usage**

```cpp
Tensor::arange(0.0, 10.0, 1.0, DType::Float32)
```

Returns:

```
Tensor
```

---

## From Vector (`from_vector`)

**Definition**

Creates a tensor by copying data from a standard C++ `std::vector`.

**Usage**

```cpp
std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
Tensor::from_vector(data, {2, 2})
```

Returns:

```
Tensor
```

---

# Shape Manipulation & Layout

Methods for changing how the underlying memory is interpreted mathematically.

---

## Reshape (`reshape`)

**Definition**

Changes the shape of the tensor without copying data, if possible. The total number of elements must remain the same.

**Usage**

```cpp
tensor.reshape({2, 6})
```

Returns:

```
Tensor
```

---

## Permute (`permute`)

**Definition**

Rearranges the dimensions of the tensor (e.g., matrix transpose). This shuffles the `shape` and `strides` without copying memory.

**Usage**

```cpp
tensor.permute({1, 0})
```

Returns:

```
Tensor
```

---

## In-Place Transpose (`t_`)

**Definition**

Performs a fast, in-place 2D matrix transpose. Only supports 2D tensors.

**Usage**

```cpp
tensor.t_()
```

Returns:

```
Tensor&
```

---

## Squeeze (`squeeze`)

**Definition**

Removes all dimensions of size 1 from the tensor's shape. (e.g., `[1, 3, 1, 4]` becomes `[3, 4]`).

**Usage**

```cpp
tensor.squeeze()
```

Returns:

```
Tensor
```

---

## Unsqueeze (`unsqueeze`)

**Definition**

Inserts a dimension of size 1 at the specified index.

**Usage**

```cpp
tensor.unsqueeze(0)
```

Returns:

```
Tensor
```

---

## Flatten (`flatten`)

**Definition**

Reshapes the entire tensor into a 1-D array containing all elements.

**Usage**

```cpp
tensor.flatten()
```

Returns:

```
Tensor
```

---

# Memory & Graph Management

---

## Clone (`clone`)

**Definition**

Performs a **deep copy** of the tensor's memory. It guarantees that the returned tensor is packed tightly into a C-contiguous layout.

**Usage**

```cpp
tensor.clone()
```

Returns:

```
Tensor
```

---

## Detach (`detach`)

**Definition**

Creates a new logical view of the SAME physical memory, but breaks the autograd graph by setting `requires_grad` to `false`.

**Usage**

```cpp
tensor.detach()
```

Returns:

```
Tensor
```

---

# Autograd & Gradients

---

## Backward (`backward`)

**Definition**

Triggers the reverse-mode auto-differentiation engine. Computes and accumulates gradients for this tensor and all its dependencies.

**Usage**

```cpp
tensor.backward()
```

Returns:

```
void
```

---

## Zero Grad (`zero_grad`)

**Definition**

Clears the gradient accumulation buffer. Necessary before calling `.backward()` in a new training loop so gradients don't double up.

**Usage**

```cpp
tensor.zero_grad()
```

Returns:

```
void
```

---
