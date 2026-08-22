# Batch Normalization (`BatchNorm`)

Batch Normalization stabilizes and accelerates training by normalizing activations per-channel using the statistics of the current mini-batch, then re-scaling them with a pair of learnable parameters.

Unlike [LayerNorm](Norm.md), which normalizes across the features of each individual sample, BatchNorm normalizes **across the batch** for each channel — which is why it behaves differently in training and inference.

---

## Definition

`BatchNorm` accepts both 2D and 4D inputs and picks the reduction axes accordingly:

- **2D input `[N, C]`** — normalizes over `N` for each channel `C`
- **4D input `[N, C, H, W]`** — normalizes over `(N, H, W)` for each channel `C`

### Constructor

```cpp
BatchNorm(int num_features, double eps = 1e-5, double momentum = 0.1);
```

| Argument | Meaning |
|---|---|
| `num_features` | Number of channels `C` |
| `eps` | Added to the variance before the square root, for numerical stability |
| `momentum` | Blend factor used when updating the running statistics |

### Members

| Member | Kind | Shape | Description |
|---|---|---|---|
| `gamma` | learnable | `[C]` | Scale applied after normalization |
| `beta` | learnable | `[C]` | Shift applied after normalization |
| `running_mean` | buffer | `[C]` | Inference-time mean, updated during training |
| `running_var` | buffer | `[C]` | Inference-time variance, updated during training |
| `training` | flag | — | Selects batch statistics vs. running statistics |

`running_mean` and `running_var` are **not** trained by gradient descent; they are updated during the forward pass by blending in the current batch statistics with the `momentum` factor.

---

## Modes

### Training (`train()`)

Normalizes with the statistics of the current batch and updates the running buffers. Gradients flow to `gamma`, `beta`, and the input through the `GradBatchNorm` node.

### Inference (`eval()`)

Normalizes with the accumulated `running_mean` / `running_var`, so a single sample produces the same output regardless of what else is in the batch.

> **Important:** forgetting to call `eval()` before validating makes batches of size 1 produce degenerate output, since the batch variance collapses to zero.

---

## Usage

```cpp
// One BatchNorm per convolution output: 16 channels
BatchNorm bn(16);

// --- Training ---
bn.train();
Tensor out = bn(conv_out);        // [N, 16, H, W]

// --- Inference ---
bn.eval();
Tensor pred = bn(conv_out);       // uses running_mean / running_var
```

Because `gamma` and `beta` are trainable, pass them to the optimizer:

```cpp
std::vector<Tensor*> params = { &bn.gamma, &bn.beta };
```

### Returns

* `Tensor` (same shape as the input)

---

## Gradient Node (`GradBatchNorm`)

The backward pass is registered automatically when the input requires gradients. The node caches the centred input `x - mean` and the inverse standard deviation `1 / sqrt(var + eps)` from the forward pass, and computes the gradients for the input, `gamma`, and `beta` together in a single backward call.

See [Autograd](Autograd.md) for how `GradFn` nodes are linked into the graph.
