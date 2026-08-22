# Functional API (`functional`) & Upsampling

The `functional` namespace holds stateless operations — no weights, no buffers, nothing to construct. They are the building blocks you call directly inside a custom `forward()`, and the implementations that the stateful modules delegate to.

Do not confuse it with `torch::nn::functional`, which is a thin convenience alias exposing `relu` and `sigmoid` (see [nn](nn.md)).

---

## Softmax (`functional::softmax`)

### Definition

```cpp
Tensor functional::softmax(const Tensor& x, int dim = -1);
```

Converts raw scores into a probability distribution along `dim`, so the values are positive and sum to 1.

The implementation is the **numerically stable** form: it subtracts `max(x, dim)` before exponentiating, so large logits cannot overflow `exp`. Negative `dim` values index from the end, `-1` meaning the last dimension.

Every step is built from differentiable core ops (`max`, `exp`, `sum`, `unsqueeze`), so the result flows through autograd without a dedicated gradient node.

### Usage

```cpp
// Class probabilities over the last dimension
Tensor probs = functional::softmax(logits, -1);

// Attention weights over the key axis
Tensor attn_w = functional::softmax(scores, -1);
```

### Returns

* `Tensor` (same shape as the input)

> For classification training, prefer `Loss::CrossEntropy` on raw logits — it fuses the log-softmax and is more stable than softmax followed by a log. See [Loss](Loss.md).

---

## Embedding Lookup (`functional::embedding`)

```cpp
Tensor functional::embedding(const Tensor& weight,
                             const Tensor& indices,
                             int padding_idx = -1);
```

Stateless row lookup into an embedding table. Documented in full in [Embedding](Embedding.md).

---

## Interpolation (`functional::interpolate`)

### Definition

```cpp
Tensor functional::interpolate(const Tensor& input,
                               const std::vector<int>& size = {},
                               const std::vector<float>& scale_factor = {},
                               InterpolateMode mode = InterpolateMode::Nearest,
                               bool align_corners = false);
```

Resamples the spatial dimensions of a tensor to a new resolution.

| Argument | Meaning |
|---|---|
| `size` | Explicit output spatial size, e.g. `{64, 64}` |
| `scale_factor` | Multiplicative resize factor |
| `mode` | Interpolation rule, see the table below |
| `align_corners` | Whether the corner pixels of input and output are aligned when computing source coordinates |

### `InterpolateMode`

```cpp
enum class InterpolateMode { Nearest, Linear, Bilinear, Bicubic, Trilinear };
```

**Currently implemented:** `Nearest` and `Bilinear`. Any other enumerator is dispatched as `bilinear` — `Linear`, `Bicubic`, and `Trilinear` are reserved for future kernels and should not be relied on yet.

### Argument rules

The function validates its arguments and throws `std::invalid_argument` when:

- neither `size` nor `scale_factor` is given
- **both** are given
- `scale_factor` is given at all — **it is not currently supported; pass `size` instead**

### Usage

```cpp
// Nearest-neighbour resize to an explicit size
Tensor small = functional::interpolate(img, {14, 14});

// Bilinear upsample, corners aligned
Tensor big = functional::interpolate(
    img, {56, 56}, {}, InterpolateMode::Bilinear, true);
```

### Returns

* `Tensor` with the spatial dimensions replaced by `size`

The underlying core operation is also callable directly, taking `size_t` extents and a mode string:

```cpp
Tensor interpolate(const Tensor& input,
                   const std::vector<size_t>& output_size,
                   const std::string& mode = "nearest",
                   bool align_corners = false);
```

---

## Upsample Module (`Upsample`)

### Definition

The `Module` wrapper around `functional::interpolate`, for use inside a `Sequential` or as a named member of a model.

```cpp
Upsample(std::vector<int> size = {},
         std::vector<float> scale_factor = {},
         InterpolateMode mode = InterpolateMode::Nearest,
         bool align_corners = false);
```

Its constructor arguments are stored and replayed on every forward call. Upsampling is a pure routing operation, so `parameters()` returns an empty vector — there is nothing to train.

Input is typically `[Batch, Channels, Height, Width]` for the 2D modes.

### Usage

```cpp
// Fixed-output-size upsampling stage in a decoder
Upsample up({32, 32}, {}, InterpolateMode::Bilinear, false);

Tensor out = up(feature_map);   // [B, C, 16, 16] -> [B, C, 32, 32]
```

### Returns

* `Tensor` (resampled)

> `Upsample` interpolates with a fixed rule and has no weights. If you want the network to *learn* how to upsample, use [`ConvTranspose2d`](ConvTranspose.md) instead.
