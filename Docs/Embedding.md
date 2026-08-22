# Embedding (`nn::Embedding`)

An embedding layer is a learnable lookup table that maps integer indices — token IDs, categorical features, positions — to dense vectors. It is the first layer of nearly every language model, including the [GPT-2 implementation](GPT2.md) shipped with CTensor.

Conceptually it is a one-hot matrix multiply, but implemented as a direct row lookup (`embedding_lookup`) so the cost is `O(indices)` rather than `O(indices × vocab)`.

---

## Definition

```cpp
nn::Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1);
```

| Argument | Meaning |
|---|---|
| `num_embeddings` | Size of the table — the vocabulary size |
| `embedding_dim` | Width of each vector |
| `padding_idx` | Optional row that is zero-initialized; `-1` (default) disables it |

The `weight` tensor has shape `[num_embeddings, embedding_dim]`, is initialized from a standard normal distribution `N(0, 1)`, and has `requires_grad` enabled on construction. If `padding_idx >= 0`, that row is zeroed after initialization.

Passing a `padding_idx` outside `[-1, num_embeddings)` throws `std::invalid_argument`.

---

## Forward Pass

Input is a tensor of integer indices of any shape; output appends the embedding dimension:

```
indices [B, S]  ->  output [B, S, embedding_dim]
```

Indices are normally `DType::Int32`.

### Usage

```cpp
// A 50257-token vocabulary mapped to 768-dimensional vectors
nn::Embedding wte(50257, 768);

Tensor token_ids = Tensor::from_vector(ids, {1, S}, DType::Int32);
Tensor x = wte(token_ids);            // [1, S, 768]

// Learned positional embeddings, added to the token embeddings
nn::Embedding wpe(1024, 768);
Tensor pos = Tensor::arange(0, (double)S, 1, DType::Int32);
x = x + wpe(pos);
```

### Returns

* `Tensor` of shape `indices.shape() + [embedding_dim]`

---

## Parameters

`Embedding` exposes its table through both parameter APIs:

```cpp
auto p     = wte.parameters();             // { &weight }
auto named = wte.named_parameters("wte."); // { {"wte.weight", &weight} }
```

That naming is what allows the table to round-trip through [safetensors](Serialization.md) under the same key a PyTorch checkpoint uses.

---

## Functional Form (`functional::embedding`)

### Definition

```cpp
Tensor functional::embedding(const Tensor& weight,
                             const Tensor& indices,
                             int padding_idx = -1);
```

The stateless equivalent, for when you already hold the weight matrix — for example, when a model ties its input embedding to its output projection. It forwards to the core `embedding_lookup(weight, indices)` operation.

> **Note:** `padding_idx` is accepted for API compatibility but is not applied by the functional form. Zeroing the padding row is done once at construction time by the `Embedding` module.

### Usage

```cpp
Tensor x = functional::embedding(my_weight, token_ids);
```

### Returns

* `Tensor`

---

## Weight Tying

Tying the output projection to the input table is a plain matrix multiply against the transposed weight — no extra parameter is created:

```cpp
Tensor flat   = x.contiguous().reshape({B * S, n_embd});
Tensor logits = matmul(flat, wte.weight.permute({1, 0}));
```

This is exactly what `GPT2Model::forward` does to produce logits.
