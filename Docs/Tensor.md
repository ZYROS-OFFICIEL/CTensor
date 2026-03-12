
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