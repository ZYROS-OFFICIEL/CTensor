# CPU Backend & Kernel Dispatch

Every tensor operation you call — `add`, `matmul`, `relu` — is a thin front-end that forwards to the fastest kernel your machine can actually run. This page explains how that choice is made, what kernels exist, and how to call a specialized kernel directly when you want to bypass the dispatcher.

You never *have* to read this to use CTensor. It matters when you are benchmarking, adding an operation, or debugging a numerical difference between two machines.

---

## Runtime Dispatch

### How a kernel is chosen

Selection happens at **runtime**, not compile time. On first use, a static `DispatchTable` is built:

1. Every `[operation][dtype]` slot is filled with the portable OpenMP-threaded kernel (the `*_mp` family in `opsmp.h`) — this always works on any x86-64 CPU.
2. If `__builtin_cpu_supports("avx2")` reports AVX2, the `Float32` and `Double64` slots are overwritten with AVX2 kernels.
3. If `__builtin_cpu_supports("avx512f")` reports AVX-512F, those slots are overwritten again with AVX-512 kernels.

Because the whole binary is compiled once and the check is a runtime CPUID query, the same executable runs correctly on a machine without AVX-512 — it simply falls back a tier. Non-x86 or non-GCC/Clang builds report no features and use the OpenMP path throughout.

### Coverage by tier

| Kernel family | Dtypes accelerated | Fallback |
|---|---|---|
| Binary ops (`add`, `sub`, `mul`, `div`, `pow`, `matmul`) | `Float32`, `Double64` | `*_mp` |
| Comparisons (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) | `Float32` (AVX2/AVX-512) | `*_mp` |
| Unary ops & activations | `Float32`, `Double64` | `*_mp` |
| Reductions (`sum`, `mean`, `max`, `min`) | `Float32`, `Double64`, **contiguous tensors only** | `*_mp` |
| Integer and boolean dtypes | — | `*_mp` always |

Two details worth knowing:

- **Reductions only take the SIMD path on flat (contiguous) tensors.** A permuted or sliced view falls back to the threaded scalar kernel. Call `.contiguous()` first if a reduction is on your hot path.
- **`gelu` and `silu` have AVX2 kernels but no AVX-512 specialization**, so on an AVX-512 machine they still run the AVX2 version.

### Checking what your CPU supports

`src/avx_flag.cpp` is a standalone probe:

```
CPU feature detection:
  AVX2     : YES
  AVX-512F : NO
```

---

## Kernel Naming

Every specialized kernel follows `<op>_<isa>_<dtype>`:

```
add_avx2_f32      relu_avx512_f32      matmul_avx2_d64      sum_avx512_d64
```

| Component | Values |
|---|---|
| `<isa>` | `avx2`, `avx512`, or `mp` (portable OpenMP) |
| `<dtype>` | `f32` (`Float32`), `d64` (`Double64`) |

You can call these directly if you know the machine and dtype, skipping the dispatch indirection:

```cpp
#include "cpu.h"

Tensor c = add_avx2_f32(a, b);   // undefined behaviour if the CPU lacks AVX2
```

> **Warning:** direct calls do no feature checking. Guard them yourself, or use the dispatched front-end.

---

## Fused Kernels

Fused kernels compute a multi-step expression in a single pass over memory. Elementwise work is memory-bound, so evaluating `a*b + c` as one kernel rather than three separate ops removes two round-trips through RAM and the temporaries that go with them.

They live under `src/cpu/AVX2/Fused_kernels/` and `src/cpu/AVX512/Fused_kernels/`, in `f32` and `d64` variants each. Argument order always follows the mathematical form.

### Multiply-accumulate

| Kernel | Computes |
|---|---|
| `fma_*(a, b, c)` | `a * b + c` |
| `fms_*(a, b, c)` | `a * b - c` |
| `nfma_*(a, b, c)` | `-(a * b) + c` |
| `mul_add_*(a, b, c)` | `a * b + c` |
| `add_scale_*(a, b, scale)` | `(a + b) * scale` |

### Add-then-activate

| Kernel | Computes |
|---|---|
| `add_relu_*(a, b)` | `relu(a + b)` |
| `add_sigmoid_*(a, b)` | `sigmoid(a + b)` |
| `add_tanh_*(a, b)` | `tanh(a + b)` |
| `add_exp_*(a, b)` | `exp(a + b)` |
| `add_ln_*(a, b)` | `ln(a + b)` |

### Unary compositions

| Kernel | Computes |
|---|---|
| `exp_neg_*(a)` | `exp(-a)` |
| `ln_relu_*(a)` | `ln(relu(a))` |
| `sigmoid_ln_*(a)` | `sigmoid(ln(a))` |

### Neural-network blocks

| Kernel | Computes |
|---|---|
| `gelu_*(a)` | GELU activation |
| `silu_*(a)` | SiLU / Swish activation |
| `swiglu_*(a, b)` | `silu(a) * b` — the SwiGLU gate used in modern transformers |
| `layer_norm_*(x, weight, bias, eps)` | Full LayerNorm in one pass over the row |
| `bias_add_relu_*(x, bias)` | `relu(x + bias)` — the standard `Linear` epilogue |
| `bias_add_gelu_*(x, bias)` | `gelu(x + bias)` |
| `scale_shift_*(x, scale, shift)` | `x * scale + shift` |

> Fused kernels are **not** wired into the automatic dispatcher — call them explicitly. They also do not build autograd nodes, so use them for inference and hand-written backward passes, not inside a graph you intend to differentiate.

### Usage

```cpp
#include "cpu.h"

// One pass instead of three
Tensor y = fma_avx512_f32(a, b, c);

// Fused LayerNorm
Tensor n = layer_norm_avx2_f32(x, weight, bias, 1e-5f);
```

---

## Type Dispatch Macro (`DISPATCH_ALL_TYPES`)

### Definition

When you write an operation that must work for every dtype, `core/dispatch.h` turns a runtime `DType` into a compile-time `scalar_t` type alias:

```cpp
DISPATCH_ALL_TYPES(tensor.dtype(), "my_op", [&] {
    scalar_t* p = (scalar_t*)tensor.impl->data->data.get();
    // ... typed code, instantiated once per dtype ...
});
```

The macro expands into a `switch` covering `Float32`, `Int32`, `Double64`, `UInt8/16/32/64`, `Int8/16/64`, and `Bool`, and throws `std::runtime_error("<name>: unsupported dtype")` for anything else. The name string you pass is what appears in that message.

---

## Threading

The portable `*_mp` kernels are parallelized with OpenMP; the library is compiled with `-fopenmp` and links `gomp`. Thread count follows the usual environment variable:

```bash
OMP_NUM_THREADS=8 ./build/trainmnist
```

Compilation flags per target are set in `CMakeLists.txt`: the AVX2 objects build with `-O3 -mavx2 -mfma -fopenmp`, the AVX-512 objects with `-O3 -mavx512f -mavx512dq`. Only those translation units get the ISA flags, which is what keeps the rest of the binary runnable on older CPUs. See [Build & Test](Build.md).

---

## GPU Kernels

`src/gpu/cuda/` contains CUDA kernel headers for `f32`, `d64`, `f16`, `int8`, and `int64`, including fused variants. They are **not** compiled by the default CMake configuration and the runtime dispatcher does not route to them yet. `Device(DeviceType::CUDA, 0)` exists in the device layer (see [Devices](Device.md)), but the CPU path is the supported one today.
