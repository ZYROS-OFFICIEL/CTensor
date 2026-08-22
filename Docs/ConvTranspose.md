# Transposed Convolution (`ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`)

Transposed convolution — sometimes called *deconvolution* or *fractionally-strided convolution* — is the gradient of a normal convolution with respect to its input, used as a forward operation. It **increases** spatial resolution instead of reducing it, which makes it the standard upsampling block in decoders, autoencoders, GAN generators, and segmentation heads.

Unlike [`Upsample`](Functional.md), which interpolates with a fixed rule, transposed convolution *learns* how to upsample.

---

## Output Shape

For each spatial dimension:

```
out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
```

`output_padding` adds extra rows/columns on one side only. It exists to disambiguate the cases where several input sizes map to the same convolution output size; it never participates in the convolution itself.

---

## ConvTranspose1d

### Definition

Applies a 1D transposed convolution over an input of shape `[Batch, in_channels, Length]`.

```cpp
ConvTranspose1d(int in_c, int out_c, int k,
                int s = 1, int p = 0, int op = 0,
                DType dt = DType::Float32);
```

| Argument | Meaning |
|---|---|
| `in_c` / `out_c` | Input and output channel counts |
| `k` | Kernel size |
| `s` | Stride (the upsampling factor) |
| `p` | Padding removed from the output |
| `op` | Extra output padding |
| `dt` | Parameter dtype |

### Usage

```cpp
// Double the length: 1 -> 8 channels, kernel 4, stride 2, padding 1
ConvTranspose1d up(1, 8, 4, 2, 1);

Tensor out = up(input);   // [B, 1, L] -> [B, 8, 2L]
```

### Returns

* `Tensor` of shape `[Batch, out_channels, L_out]`

---

## ConvTranspose2d

### Definition

Applies a 2D transposed convolution over an input of shape `[Batch, in_channels, Height, Width]`. Height and width are configured independently.

```cpp
ConvTranspose2d(int in_c, int out_c, int kh, int kw = -1,
                int sh = 1, int sw = 1,
                int ph = 0, int pw = 0,
                int oph = 0, int opw = 0,
                DType dt = DType::Float32);
```

Passing `kw = -1` reuses `kh`, giving a square kernel.

### Usage

```cpp
// Classic 2x upsampling block used in decoders
ConvTranspose2d up(64, 32, 4, 4, 2, 2, 1, 1);

Tensor out = up(feature_map);   // [B, 64, 16, 16] -> [B, 32, 32, 32]

// Square kernel shorthand
ConvTranspose2d simple(64, 32, 3);
```

### Returns

* `Tensor` of shape `[Batch, out_channels, H_out, W_out]`

---

## ConvTranspose3d

### Definition

Applies a 3D transposed convolution over volumetric input of shape `[Batch, in_channels, Depth, Height, Width]`, with per-axis kernel, stride, padding, and output padding.

```cpp
ConvTranspose3d(int in_c, int out_c, int kd, int kh, int kw,
                int sd = 1, int sh = 1, int sw = 1,
                int pd = 0, int ph = 0, int pw = 0,
                int opd = 0, int oph = 0, int opw = 0,
                DType dt = DType::Float32);
```

### Usage

```cpp
ConvTranspose3d up(16, 8, 2, 2, 2, 2, 2, 2);

Tensor out = up(volume);   // [B, 16, D, H, W] -> [B, 8, 2D, 2H, 2W]
```

### Returns

* `Tensor` of shape `[Batch, out_channels, D_out, H_out, W_out]`

---

## Parameters & Autograd

All three layers inherit from `Module`, hold a `weight` and a `bias` tensor, and expose them through `parameters()`:

```cpp
ConvTranspose2d up(64, 32, 4, 4, 2, 2, 1, 1);
optim::AdamW optimizer(up.parameters(), 1e-3);
```

The backward pass is handled by the `GradConvTranspose1d` / `GradConvTranspose2d` / `GradConvTranspose3d` nodes, which register `input`, `weight`, and `bias` as graph parents and therefore produce gradients for all three. See [Conv](Conv.md) for the forward-convolution counterpart and [Autograd](Autograd.md) for graph mechanics.
