# Pooling Layers

Pooling layers are used to progressively reduce the spatial dimensions (downsample) of tensors. This decreases computational load, reduces memory usage, and helps the network achieve spatial invariance to small translations in the input.

All pooling layers inherit from the base `Module` class but contain no learnable parameters.

---

## Max Pooling (`MaxPool1d`, `MaxPool2d`, `MaxPool3d`)

### Definition
Applies a 1D, 2D, or 3D max pooling over an input signal composed of several input planes.

- Slides a window of `kernel_size` over the input
- Outputs the **maximum value** within that window
- During the `.backward()` pass, gradients are routed only to the index that held the maximum value

If `stride` is not explicitly provided (or set to `-1`), it automatically defaults to `kernel_size`.

### Variants
- **MaxPool1d**: For sequential or 1D spatial data  
  - Expected shape: `[Batch, Channels, Length]`
- **MaxPool2d**: For images or 2D spatial data  
  - Expected shape: `[Batch, Channels, Height, Width]`
- **MaxPool3d**: For volumetric or 3D spatial data  
  - Expected shape: `[Batch, Channels, Depth, Height, Width]`

### Usage

```cpp
// Initialize MaxPool2d with kernel_size = 2x2, stride = 2x2, padding = 0
MaxPool2d max_pool(2, 2, 2, 2, 0, 0);

// For simpler square kernels, if stride isn't given, it equals kernel_size
MaxPool2d quick_pool(2, 2);

// Forward pass
Tensor output = quick_pool(input_tensor);
````

### Returns

* `Tensor` (Spatially downsampled tensor)

---

## Average Pooling (`AvgPool1d`, `AvgPool2d`, `AvgPool3d`)

### Definition

Applies a 1D, 2D, or 3D average pooling over an input signal.

* Computes the **mean** of all elements within the `kernel_size` window
* During the `.backward()` pass, gradients are distributed evenly across all elements in the window

Like max pooling, if `stride` is not provided (or set to `-1`), it automatically defaults to `kernel_size`.

### Variants

* **AvgPool1d**: For sequential or 1D spatial data
* **AvgPool2d**: For images or 2D spatial data
* **AvgPool3d**: For volumetric or 3D spatial data

### Usage

```cpp
// Initialize AvgPool2d with kernel_size = 3x3, stride = 1, padding = 1
AvgPool2d avg_pool(3, 3, 1, 1, 1, 1);

// Initialize simple 2x2 average pool with default stride (2x2)
AvgPool2d simple_avg(2, 2);

// Forward pass
Tensor output = simple_avg(input_tensor);
```

### Returns

* `Tensor` (Spatially downsampled tensor)
