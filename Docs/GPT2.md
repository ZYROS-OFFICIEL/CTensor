# GPT-2 (`GPT2Model`, `GPT2Tokenizer`)

CTensor ships a complete GPT-2 implementation: the transformer itself, a byte-level BPE tokenizer, and a loader that reads the official Hugging Face `model.safetensors` weights. Together they run real text generation with the pretrained 124M checkpoint — no Python at any step.

Everything is built from the library's own primitives ([Embedding](Embedding.md), [LayerNorm](Norm.md), `Linear`, `Dropout`, `bmm`, `gelu`, `softmax`), so it doubles as a reference for assembling your own transformer.

---

## Configuration (`GPT2Config`)

### Definition

```cpp
struct GPT2Config {
    int n_layer;                    // number of transformer blocks
    int n_head;                     // attention heads per block
    int n_embd;                     // model width; must be divisible by n_head
    int vocab_size;                 // token vocabulary size
    int n_positions;                // maximum context length
    double dropout = 0.1;
    double layer_norm_eps = 1e-5;
};
```

Two presets are provided:

| Factory | Shape |
|---|---|
| `GPT2Config::toy(vocab_size)` | 2 layers, 2 heads, width 32, context 64 — fast enough for unit tests |
| `GPT2Config::gpt2_small_124M()` | 12 layers, 12 heads, width 768, vocab 50257, context 1024 — the released GPT-2 small |

### Usage

```cpp
GPT2Config cfg = GPT2Config::gpt2_small_124M();
GPT2Model model(cfg);
```

---

## Causal Mask (`make_causal_mask`)

### Definition

```cpp
Tensor make_causal_mask(size_t S);
```

Builds the `[S, S]` additive mask that stops a position from attending to later positions: entries above the diagonal are `-1e9`, everything else is `0`. Because it is additive, softmax drives the masked entries to zero.

`GPT2Model::forward` rebuilds it for each sequence length, so you rarely call it directly.

### Returns

* `Tensor` of shape `[S, S]`, `Float32`

---

## Architecture

The model is a pre-norm transformer — LayerNorm sits *before* each sublayer, and each sublayer is wrapped in a residual connection:

```
x = x + attn(ln_1(x), mask)
x = x + mlp(ln_2(x))
```

### `GPT2Attention`

Multi-head causal self-attention. Projects the input with `q_proj`, `k_proj`, `v_proj`, folds the heads into the batch dimension (`[B, S, n_embd] → [B·H, S, head_dim]`) so the scores can be computed with a single `bmm`, scales by `1/sqrt(head_dim)`, adds the causal mask, applies `softmax` and `attn_dropout`, then merges the heads back and applies `out_proj` followed by `resid_dropout`.

The constructor throws `std::invalid_argument` if `n_embd` is not divisible by `n_head`.

### `GPT2MLP`

The position-wise feed-forward network: `c_fc` widens to `4 × n_embd`, `gelu` activates, `c_proj` projects back, `resid_dropout` regularizes.

### `GPT2Block`

One `ln_1` + `GPT2Attention` + `ln_2` + `GPT2MLP` unit with its two residual connections.

### `GPT2Model`

- `wte` — token embedding `[vocab_size, n_embd]`
- `wpe` — learned positional embedding `[n_positions, n_embd]`
- `drop` — embedding dropout
- `blocks` — `n_layer` `GPT2Block`s
- `ln_f` — final LayerNorm

The output projection is **tied** to `wte`: logits are produced by multiplying the final hidden states by the transposed embedding matrix, so there is no separate `lm_head` parameter.

### Initialization

`init_weights()` reproduces the GPT-2 scheme: all weights are drawn from `N(0, 0.02)`, biases are zeroed, LayerNorm weights are set to 1 and biases to 0, and the two residual-output projections (`attn.out_proj`, `mlp.c_proj`) use a reduced standard deviation of `0.02 / sqrt(2 · n_layer)` to keep the residual stream from growing with depth.

---

## Forward Pass

### Definition

```cpp
Tensor GPT2Model::forward(const Tensor& token_ids);
```

Takes integer token IDs of shape `[Batch, SeqLen]`, adds token and positional embeddings, runs every block under the causal mask, applies the final LayerNorm, and projects to the vocabulary.

### Usage

```cpp
Tensor token_ids = Tensor::from_vector(ids, {1, S}, DType::Int32);
Tensor logits = model(token_ids);      // [1, S, vocab_size]
```

### Returns

* `Tensor` of shape `[Batch, SeqLen, vocab_size]` (raw logits)

---

## Text Generation (`generate`)

### Definition

```cpp
std::vector<int> GPT2Model::generate(const std::vector<int>& prompt_ids,
                                     int max_new_tokens,
                                     double temperature = 1.0);
```

Autoregressively appends `max_new_tokens` tokens to the prompt. Each step re-runs the forward pass on the last `n_positions` tokens (the context window slides once the prompt outgrows it), reads the logits at the final position, and samples the next ID.

| `temperature` | Behaviour |
|---|---|
| `<= 1e-6` | Greedy — always takes the arg-max token, fully deterministic |
| `1.0` | Samples from the unmodified distribution |
| `< 1.0` | Sharpens the distribution: more conservative |
| `> 1.0` | Flattens it: more diverse, less coherent |

`generate()` calls `eval()` itself, so dropout is disabled. Sampling uses a fixed seed (`mt19937(42)`), which makes non-greedy runs reproducible within a process.

There is no KV cache — each step recomputes the whole window, so generation is `O(n²)` in the number of tokens.

### Usage

```cpp
std::vector<int> ids = model.generate(prompt_ids, 20, 0.0);   // greedy
std::vector<int> ids = model.generate(prompt_ids, 50, 0.8);   // sampled
```

### Returns

* `std::vector<int>` — the prompt followed by the newly generated IDs

---

## Loading Hugging Face Weights (`gpt2_io::load_huggingface`)

### Definition

```cpp
safetensors::LoadReport gpt2_io::load_huggingface(const std::string& path,
                                                  GPT2Model& model,
                                                  bool strict = true);
```

Reads an official GPT-2 `model.safetensors` and copies it into a `GPT2Model`, translating the two places where the reference implementation differs from CTensor:

1. **`Conv1D` weights are transposed.** The original GPT-2 uses TensorFlow-style `Conv1D` layers whose weights are stored `[in, out]`, while `Linear` expects `[out, in]`. Every `c_attn`, `c_proj`, and `c_fc` weight is transposed on the way in.
2. **The fused QKV projection is split.** GPT-2 packs query, key, and value into one `attn.c_attn` matrix of width `3 · n_embd`; the loader slices it into the separate `q_proj`, `k_proj`, and `v_proj` weights and biases.

Keys that carry no information for this implementation — `attn.bias` and `attn.masked_bias` (precomputed masks, rebuilt by `make_causal_mask`) and `lm_head.weight` (tied to `wte`) — are consumed and ignored.

Any checkpoint key the loader never touches lands in `report.unexpected`. With `strict = true` that throws; a missing *expected* key always throws, regardless of `strict`. Shape disagreements throw with the offending parameter name.

### Usage

```cpp
GPT2Config cfg = GPT2Config::gpt2_small_124M();
GPT2Model model(cfg);

auto report = gpt2_io::load_huggingface("model.safetensors", model, true);
std::cout << "unexpected keys: " << report.unexpected.size() << "\n";
```

### Returns

* `safetensors::LoadReport` — see [Serialization](Serialization.md)

---

## Tokenizer (`GPT2Tokenizer`)

### Definition

```cpp
static GPT2Tokenizer GPT2Tokenizer::from_files(const std::string& vocab_json_path,
                                               const std::string& merges_txt_path);

std::vector<int> encode(const std::string& text) const;
std::string      decode(const std::vector<int>& ids) const;
```

A faithful byte-level BPE tokenizer, matching the reference GPT-2 implementation:

- **Byte-to-unicode mapping.** All 256 byte values are mapped to printable unicode codepoints, so any binary input is representable and no `<unk>` token is ever needed.
- **Pre-tokenization.** Text is split UTF-8-aware into contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), letter runs, digit runs, symbol runs, and whitespace, with a leading space attached to the following word — GPT-2's distinctive `Ġword` tokens.
- **BPE merging.** Adjacent symbol pairs are merged repeatedly, always taking the pair with the lowest rank in `merges.txt`, until no ranked pair remains.

`from_files` parses `vocab.json` with the built-in [JSON parser](Serialization.md) and reads `merges.txt`, skipping its version header line and using each line's position as its merge rank. It throws if either file cannot be opened or `vocab.json` is not a JSON object. `encode` throws if a merged symbol is absent from the vocabulary.

`decode` reverses the process: tokens are concatenated and the unicode codepoints mapped back to raw bytes.

### Usage

```cpp
GPT2Tokenizer tok = GPT2Tokenizer::from_files("vocab.json", "merges.txt");

std::vector<int> ids = tok.encode("The capital of France is");
std::string text     = tok.decode(ids);
```

### Returns

| Method | Return Type |
|---|---|
| `from_files` | `GPT2Tokenizer` |
| `encode` | `std::vector<int>` |
| `decode` | `std::string` |

---

## End-to-End Example

The three files come from the [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) repository:

- `model.safetensors`
- `vocab.json`
- `merges.txt`

```cpp
#include "core.h"
#include "neuralnet.h"
#include "models/gpt2.h"
#include "models/gpt2_tokenizer.h"

int main() {
    GPT2Config cfg = GPT2Config::gpt2_small_124M();
    GPT2Model model(cfg);

    gpt2_io::load_huggingface("model.safetensors", model, true);
    GPT2Tokenizer tok = GPT2Tokenizer::from_files("vocab.json", "merges.txt");

    std::vector<int> prompt = tok.encode("The capital of France is");

    model.eval();
    std::vector<int> out = model.generate(prompt, 20, 0.0);

    std::cout << tok.decode(out) << "\n";
    return 0;
}
```

Build and run the bundled version of this program with:

```bash
cmake --build build --target test_gpt2_huggingface
./build/test_gpt2_huggingface model.safetensors vocab.json merges.txt "The capital of France is"
```

`test_gpt2_toy` trains a tiny config from scratch and `test_gpt2_primitives` checks the individual blocks — see [Build & Test](Build.md).
