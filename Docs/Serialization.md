# Serialization (`safetensors`)

The `safetensors` namespace reads and writes the [safetensors](https://github.com/huggingface/safetensors) format: a JSON header describing every tensor, followed by one flat block of raw bytes. It is the interchange format used by Hugging Face, so a CTensor model can load weights produced by PyTorch — and vice versa — without a Python step.

Unlike the custom binary format in [Checkpoints](Checkpoint.md), which stores parameters positionally, safetensors stores them **by name**. That makes checkpoints robust to layer reordering and lets you diagnose exactly which key failed to match.

---

## File Layout

| Bytes | Contents |
|---|---|
| `0 .. 8` | Header length `N`, unsigned 64-bit little-endian |
| `8 .. 8+N` | UTF-8 JSON header, space-padded so the data section starts on an 8-byte boundary |
| `8+N .. EOF` | Raw tensor data, concatenated |

Each header entry maps a tensor name to `{"dtype": ..., "shape": [...], "data_offsets": [begin, end]}`, where the offsets are relative to the start of the data section. The reserved key `__metadata__` holds a flat string-to-string map.

The loader validates aggressively and throws `std::runtime_error` on: a header longer than the file, a non-object header, a missing `dtype`/`shape`/`data_offsets` field, a negative shape entry, out-of-range offsets, a byte length that disagrees with `shape × dtype_size`, a truncated data section, or an unsupported dtype.

### Supported dtypes

| safetensors | CTensor `DType` |
|---|---|
| `F64`, `F32`, `F16`, `BF16` | `Double64`, `Float32`, `Float16`, `BFloat16` |
| `I64`, `I32`, `I16`, `I8` | `Int64`, `Int32`, `Int16`, `Int8` |
| `U64`, `U32`, `U16`, `U8` | `UInt64`, `UInt32`, `UInt16`, `UInt8` |
| `BOOL` | `Bool` |

---

## Load Tensors (`load`)

### Definition

```cpp
std::unordered_map<std::string, Tensor> load(
    const std::string& path,
    std::unordered_map<std::string, std::string>* metadata_out = nullptr);
```

Reads every tensor into a name-keyed map, preserving the dtype recorded in the file. Pass a non-null `metadata_out` to also receive the `__metadata__` entries; string values only.

### Usage

```cpp
std::unordered_map<std::string, std::string> meta;
auto tensors = safetensors::load("model.safetensors", &meta);

Tensor& w = tensors.at("fc1.weight");
std::cout << meta["format"] << "\n";
```

### Returns

* `std::unordered_map<std::string, Tensor>`

---

## Save Tensors (`save`)

### Definition

Two overloads write a file:

```cpp
// 1. An explicit, ordered list of (name, tensor) pairs
void save(const std::string& path,
          const std::vector<std::pair<std::string, Tensor>>& tensors,
          const std::unordered_map<std::string, std::string>& metadata = {});

// 2. A model's named parameters, straight from Module::named_parameters()
void save(const std::string& path,
          const Module::NamedParams& params,
          const std::unordered_map<std::string, std::string>& metadata = {});
```

Both handle the awkward cases for you: CUDA tensors are copied to the CPU first, and non-contiguous tensors (views, slices, permutations) are made contiguous before their bytes are written, so a sliced tensor round-trips correctly.

The `NamedParams` overload skips null and uninitialized entries. Using `"__metadata__"` as a tensor name throws, as does saving an empty tensor or failing to open the file for writing.

### Usage

```cpp
// Explicit tensors, with metadata
std::vector<std::pair<std::string, Tensor>> tensors = {{"a", a}, {"b", b}};
safetensors::save("out.safetensors", tensors, {{"format", "ctensor"}});

// A whole model
safetensors::save("model.safetensors", model.named_parameters());
```

### Returns

`void`

---

## Load Into a Model (`load_into`)

### Definition

```cpp
LoadReport load_into(const std::string& path,
                     const Module::NamedParams& params,
                     bool strict = true);
```

Loads a file and copies each tensor into the matching parameter **by name**, in place — so the model keeps its existing allocation and dtype. If the file's dtype differs from the parameter's, the data is converted via `astype` before the copy.

Nothing is copied for a parameter whose shape disagrees with the file; it is reported instead.

### `LoadReport`

```cpp
struct LoadReport {
    std::vector<std::string> missing;         // parameter had no entry in the file
    std::vector<std::string> unexpected;      // file entry matched no parameter
    std::vector<std::string> shape_mismatch;  // name matched, shape did not
    bool ok() const;                          // true when all three are empty
};
```

With `strict = true` (the default) a non-empty report throws `std::runtime_error` listing every offending key. With `strict = false` the load proceeds as far as it can and hands you the report — the right choice when loading a pretrained backbone into a model with a new head.

### Usage

```cpp
MyModel model;

// Strict: any mismatch throws
auto report = safetensors::load_into("model.safetensors", model.named_parameters());

// Tolerant: inspect what did not line up
auto report = safetensors::load_into("pretrained.safetensors",
                                     model.named_parameters(),
                                     /*strict=*/false);
if (!report.ok()) {
    for (auto& k : report.missing)        std::cout << "missing: "   << k << "\n";
    for (auto& k : report.unexpected)     std::cout << "unexpected: "<< k << "\n";
    for (auto& k : report.shape_mismatch) std::cout << "shape: "     << k << "\n";
}
```

### Returns

* `LoadReport`

---

## JSON Parser (`parse_json`)

The header is parsed by a small dependency-free JSON reader in `core/json.h`, which is also used by the [GPT-2 tokenizer](GPT2.md) to read `vocab.json`.

```cpp
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object } type;
    bool b;
    double num;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const;  // nullptr if absent
};

JsonValue parse_json(const std::string& text);
```

Object keys keep their **file order** (they are stored in a vector, not a map), which is what makes offset bookkeeping in the safetensors header predictable. `find()` is a linear scan — fine for headers, not intended for hot loops.

### Usage

```cpp
JsonValue root = parse_json(text);
if (const JsonValue* v = root.find("shape")) {
    for (auto& d : v->arr) std::cout << (size_t)d.num << " ";
}
```

---

## Which Format Should I Use?

| | `checkpoints::save_weights` | `safetensors::save` |
|---|---|---|
| Keyed by | Position in the parameter vector | Name |
| Cross-framework | No | Yes (PyTorch, Hugging Face) |
| Mismatch diagnostics | Shape/size assertions | `LoadReport` with per-key detail |
| Metadata | None | Arbitrary string map |

Use safetensors for anything you intend to share or reload across code changes; see [Checkpoints](Checkpoint.md) for the lightweight in-house alternative.
