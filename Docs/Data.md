# Data Loading & I/O

The `data.h` and `tensorio` modules provide utilities to convert **C++ data structures, raw memory, and files** into `Tensor` objects.

---

## From C++ Vectors

### Definition
Converts standard C++ `std::vector` objects (flat or nested) into tensors.

### Usage
```cpp
// 1D / Flat
Tensor t1 = from_flat_vector<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});

// 2D Nested
Tensor t2 = from_2d_vector<float>({{1.0, 2.0}, {3.0, 4.0}});

// 3D Nested
Tensor t3 = from_3d_vector<float>({{{1.0}, {2.0}}, {{3.0}, {4.0}}});
```

**Returns:** `Tensor`

---

## From Raw Pointers (`from_raw_ptr`)

### Definition
Creates a `Tensor` by safely copying data directly from a raw C/C++ pointer.

### Usage
```cpp
float* my_data = get_data_from_somewhere();

Tensor t = from_raw_ptr(
    my_data,
    4,
    {2, 2},
    DType::Float32
);
```

| Parameter | Description |
|-----------|-------------|
| `data` | Pointer to raw memory |
| `size` | Total number of elements |
| `shape` | Tensor shape |
| `dtype` | Data type |

**Returns:** `Tensor`

---

## File I/O (`from_csv`, `from_binary`, `from_npy`)

### Definition
Loads tensor data stored on disk.

Supported formats:

- **CSV (Comma-Separated Values)**
- **Binary dumps**
- **NumPy `.npy` files**

### Usage
```cpp
Tensor t_csv = from_csv("data.csv", DType::Float32, true, ',');
Tensor t_bin = from_binary("weights.bin", {128, 256}, DType::Float32);
Tensor t_npy = from_npy("features.npy");
```

**Returns:** `Tensor`

---

## Image I/O (`tensorio::from_image`, `tensorio::to_image`)

### Definition
Reads or writes tensors using standard image formats such as **PNG** or **JPG**.

Images are loaded in the format:

```
[Channels, Height, Width]
```

### Usage
```cpp
// Load image
Tensor img = tensorio::from_image("input.png");

// Save tensor as image
tensorio::to_image(img, "output.jpg");
```

| Function | Return |
|--------|--------|
| `from_image` | `Tensor` |
| `to_image` | `void` |

---

# Transforms Pipeline

The `Transforme` structure allows chaining multiple preprocessing operations into a **single reusable pipeline**.

This is especially useful for **image preprocessing before neural network training or inference**.

---

## Normalize (`normalize_`)

### Definition
Applies normalization to a tensor by subtracting the **mean** and dividing by the **standard deviation**.

```
normalized = (x - mean) / std
```

This operation is appended to the transformation pipeline.

### Usage
```cpp
Transforme transform;

// Global normalization
transform.normalize_({0.5}, {0.5});

// Channel-wise normalization (RGB)
transform.normalize_(
    {0.485, 0.456, 0.406},
    {0.229, 0.224, 0.225}
);
```

**Returns:** `void`

---

## Resize (`resize_`)

### Definition
Appends a resize operation to the transformation pipeline.

Currently acts as a wrapper/placeholder for spatial interpolation.

### Usage
```cpp
Transforme transform;

transform.resize_(224, 224);
```

| Parameter | Description |
|-----------|-------------|
| `height` | Target height |
| `width` | Target width |

**Returns:** `void`

---

## Apply Transformations (`operator()`)

### Definition
Executes the entire configured transformation pipeline sequentially on an input tensor.

### Usage
```cpp
Transforme transform;

transform.resize_(224, 224);
transform.normalize_({0.5}, {0.5});

// Load image
Tensor image = tensorio::from_image("cat.png");

// Apply pipeline
Tensor processed_image = transform(image);
```

**Returns:** `Tensor`