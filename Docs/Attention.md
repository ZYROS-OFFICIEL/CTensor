# Attention (`scaled_dot_product_attention`, `nn::MultiheadAttention`)

Attention lets every position in a sequence read from every other position, weighted by learned relevance. CTensor provides a functional kernel (`scaled_dot_product_attention`) and a full multi-head module (`nn::MultiheadAttention`) built on top of it.

---

## Scaled Dot-Product Attention (`scaled_dot_product_attention`)

### Definition

Implements the core attention equation:

```
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_head)) V
```

```cpp
Tensor scaled_dot_product_attention(const Tensor& query,
                                    const Tensor& key,
                                    const Tensor& value,
                                    const Tensor& attn_mask = Tensor(),
                                    double dropout_p = 0.0);
```

Step by step:

1. Transposes the last two dimensions of `key`, whatever its rank
2. `matmul(query, keyᵀ)` produces the raw score matrix
3. Scales the scores by `1 / sqrt(head_dim)`, where `head_dim` is the last dimension of `query` — without this the softmax saturates as dimensionality grows
4. Adds `attn_mask` to the scores **if a mask was provided** (an empty `Tensor()` is skipped)
5. Applies `softmax` along the last dimension
6. Multiplies the weights by `value`

Inputs are expected as `[..., seq_len, head_dim]`; leading dimensions (batch, heads) are carried through by `matmul`.

> **Note:** `dropout_p` is accepted for API compatibility but is not currently applied inside the function. Apply dropout to the attention weights yourself if you need it — as [`GPT2Attention`](GPT2.md) does.

### Masking

Masks are **additive**, not boolean: forbidden positions carry a large negative value that softmax drives to zero. A causal mask, which stops a position from reading the future, is built with:

```cpp
Tensor mask = make_causal_mask(S);   // (col > row) * -1e9, shape [S, S]
```

### Usage

```cpp
// Unmasked (bidirectional) attention
Tensor out = scaled_dot_product_attention(q, k, v);

// Causal (autoregressive) attention
Tensor out = scaled_dot_product_attention(q, k, v, make_causal_mask(S));
```

### Returns

* `Tensor` of shape `[..., q_len, head_dim]`

---

## Multi-Head Attention (`nn::MultiheadAttention`)

### Definition

Runs several attention heads in parallel on different learned projections of the same input, then concatenates and re-projects the results. Each head works in a `head_dim = embed_dim / num_heads` subspace, so several heads cost the same as one wide one.

```cpp
nn::MultiheadAttention(int embed_dim, int num_heads, bool bias = true);
```

| Argument | Meaning |
|---|---|
| `embed_dim` | Model width. Must be divisible by `num_heads`, otherwise the constructor throws `std::invalid_argument` |
| `num_heads` | Number of parallel attention heads |
| `bias` | Whether the four internal `Linear` projections carry a bias |

The module owns four `Linear` layers — `q_proj`, `k_proj`, `v_proj`, and `out_proj` — each `embed_dim → embed_dim`.

### Forward Pass

```cpp
Tensor forward(const Tensor& query,
               const Tensor& key,
               const Tensor& value,
               const Tensor& attn_mask = Tensor());
```

Inputs are batch-first, `[Batch, SeqLen, embed_dim]`. Internally the module:

1. Projects `query`, `key`, and `value`
2. Reshapes each to `[B, S, num_heads, head_dim]` and permutes to `[B, num_heads, S, head_dim]`
3. Calls `scaled_dot_product_attention`
4. Permutes back, makes the result contiguous, reshapes to `[B, q_len, embed_dim]`
5. Applies `out_proj`

Because `query`, `key`, and `value` are separate arguments, the module covers both **self-attention** (pass the same tensor three times) and **cross-attention** (queries from the decoder, keys/values from the encoder). `key` and `value` may have a different sequence length than `query`.

### Usage

```cpp
nn::MultiheadAttention attn(512, 8);   // 8 heads of width 64

// Self-attention
Tensor out = attn(x, x, x);

// Causal self-attention
Tensor out = attn(x, x, x, make_causal_mask(x.shape()[1]));

// Cross-attention: decoder queries attend to encoder memory
Tensor out = attn(decoder_x, encoder_mem, encoder_mem);

// All four projections at once
optim::AdamW optimizer(attn.parameters(), 1e-4);
```

### Returns

* `Tensor` of shape `[Batch, q_len, embed_dim]`

---

## Choosing Between Them

| Use | When |
|---|---|
| `scaled_dot_product_attention` | You already hold projected, head-split Q/K/V — custom attention variants, fused blocks |
| `nn::MultiheadAttention` | You want the standard block with its own learnable projections |

For a worked example of a hand-rolled attention block that batches the heads into `bmm` and inserts dropout on the attention weights, see `GPT2Attention` in [GPT2](GPT2.md).
