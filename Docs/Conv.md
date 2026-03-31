# Convolutional Layers

Convolutional layers are the core building blocks of Convolutional Neural Networks (CNNs). They apply sliding filters (kernels) over the input data to extract spatial hierarchies of features, such as edges, textures, or complex patterns.

These layers are stateful and contain learnable weights (the filters) and bias parameters. The 2D and 3D implementations are highly optimized using an `im2col` (image-to-column) transformation followed by a fast Matrix Multiplication (`matmul`).

---

## Common Parameters

All convolutional layers share a similar set of configuration parameters:

- `in_channels`: Number of channels in the input image/signal.
- `out_channels`: Number of channels produced by the convolution (equivalent to the number of filters).
- `kernel_size`: The size of the sliding convolving kernel (can be a single integer for square/cubic kernels, or distinct values per dimension).
- `stride`: The step size at which the kernel slides over the input (default: 1).
- `padding`: Zero-padding added to both sides of the input (default: 0).

---

## 1D Convolution (`Conv1d`)

### Definition

Applies a 1D convolution over an input signal composed of several input planes. Typically used for sequential data, time-series, or text embeddings.

- **Expected Input Shape**: `[Batch, in_channels, Length]`  
- **Weight Shape**: `[out_channels, in_channels, kernel_size]`  

### Usage

```cpp
// Initialize Conv1d: 16 input channels, 32 output channels, kernel_size = 3, stride = 1, padding = 1
Conv1d conv1d_layer(16, 32, 3, 1, 1);

// Forward pass
Tensor output = conv1d_layer(input_tensor);
````

**Returns**: `Tensor` (Shape: `[Batch, out_channels, out_length]`)

---

## 2D Convolution (`Conv2d`)

### Definition

Applies a 2D convolution over an input signal. This is the standard layer used for processing images. It utilizes an optimized `im2col` approach for rapid forward and backward passes.

> If `kw` (kernel width) is not explicitly provided or is set to `-1`, it defaults to creating a square kernel using `kh` (kernel height).

* **Expected Input Shape**: `[Batch, in_channels, Height, Width]`
* **Weight Shape**: `[out_channels, in_channels, kernel_size_h, kernel_size_w]`

### Usage

```cpp
// Square kernel: 3 RGB input channels, 64 output channels, 3x3 kernel, stride 1, padding 1
Conv2d conv2d_square(3, 64, 3, 3, 1, 1, 1, 1);

// Forward pass
Tensor image_features = conv2d_square(image_tensor);

// Retrieve learnable parameters for the optimizer
std::vector<Tensor*> params = conv2d_square.parameters();
```

**Returns**: `Tensor` (Shape: `[Batch, out_channels, out_height, out_width]`)

---

## 3D Convolution (`Conv3d`)

### Definition

Applies a 3D convolution over an input signal. Commonly used for volumetric data (like medical CT scans) or spatio-temporal data (like video frames). Like `Conv2d`, it uses a highly optimized `im2col` algorithm.

* **Expected Input Shape**: `[Batch, in_channels, Depth, Height, Width]`
* **Weight Shape**: `[out_channels, in_channels, kernel_size_d, kernel_size_h, kernel_size_w]`

### Usage

```cpp
// 1 input channel (e.g., grayscale voxel), 16 output channels, 3x3x3 kernel, stride 1, padding 0
Conv3d conv3d_layer(1, 16, 3, 3, 3, 1, 1, 1, 0, 0, 0);

// Forward pass
Tensor volume_features = conv3d_layer(voxel_tensor);
```

**Returns**: `Tensor` (Shape: `[Batch, out_channels, out_depth, out_height, out_width]`)

```
