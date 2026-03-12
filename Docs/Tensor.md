
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

