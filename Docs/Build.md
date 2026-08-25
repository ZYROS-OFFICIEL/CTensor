# Building, Testing & Installing

CTensor builds with CMake (≥ 3.26) and a C++20 compiler. GCC or Clang on x86-64 is required — the SIMD backend uses GCC/Clang CPU intrinsics and `__builtin_cpu_supports`. OpenMP (`libgomp`) is linked unconditionally.

---

## Quick Build

```bash
cmake -B build
cmake --build build
```

That produces `build/libmyproject.a` plus every test and demo executable. Parallelize with `cmake --build build -j$(nproc)`.

### Options

| Option | Default | Effect |
|---|---|---|
| `MYPROJECT_BUILD_TESTS` | `ON` | Build the test and benchmark executables |
| `CMAKE_BUILD_TYPE` | *(empty)* | Standard CMake build type |

Library only, no tests:

```bash
cmake -B build -DMYPROJECT_BUILD_TESTS=OFF
cmake --build build
```

> The AVX2 and AVX-512 objects carry their own `-O3` flags, so those kernels are optimized even in an unspecified build type. Everything else follows `CMAKE_BUILD_TYPE` — set `-DCMAKE_BUILD_TYPE=Release` for a fair benchmark.

---

## Target Layout

The build is split into `OBJECT` libraries, bundled into one static library. The split exists so each group can carry its own compile flags.

| Object library | Sources | Flags |
|---|---|---|
| `opsmp` | Portable OpenMP kernels | — |
| `core` | Tensor, autograd, dispatch, data, transforms, JSON | — |
| `neuralnet` | Layers, losses, activations, norms, dropout, pooling, batchnorm, weight init, safetensors | — |
| `conv` | Convolution | — |
| `models` | GPT-2 model, HF loader, tokenizer | — |
| `dashboard` | Metric logging | — |
| `ops_avx2` | AVX2 kernels + fused kernels | `-O3 -mavx2 -mfma -fopenmp` |
| `ops_avx512` | AVX-512 kernels + fused kernels | `-O3 -mavx512f -mavx512dq` |
| **`myproject`** | Static library combining all of the above | `-fopenmp`, links `gomp` |

Only the two SIMD object libraries are compiled with ISA flags. The rest of the binary stays baseline x86-64, which is what allows one build to run on machines with or without AVX-512 — the choice is made at runtime. See [Backend](Backend.md).

---

## Test Executables

Every test is a standalone `main()` that asserts and returns non-zero on failure. There is no test framework and no `ctest` registration, so run them directly.

```bash
cmake --build build --target test_tensor
./build/test_tensor
```

Run the lot:

```bash
for t in build/test_*; do echo "== $t"; "$t" || echo "FAILED"; done
```

### Core

| Target | Covers |
|---|---|
| `test_tensor` | Tensor creation, shapes, views, dtypes |
| `test_ops` | Arithmetic, comparisons, reductions |
| `test_autograd` | Backward passes and graph construction |
| `test_data` | Datasets and data handling |
| `transformetest` | Transforms |

### Neural network

| Target | Covers |
|---|---|
| `layer_test` | `Linear`, `Flatten` |
| `loss_test` | Loss functions |
| `test_relu` | Activations |
| `test_norm` | Norms and LayerNorm |
| `test_batchnorm` | BatchNorm, both modes |
| `test_dropout` | Dropout, train vs. eval |
| `test_pooling` | Max/avg pooling |
| `test_conv` | Convolution |
| `test_metrics` | Accuracy, F1, and friends |
| `test_optimizers` | SGD, Adam, AdamW, and the rest |
| `test_dataloader` | Batching and shuffling |
| `test_weights_init` | Initialization schemes |
| `test_checks` | Debug/validation helpers |
| `test_nn` | The `torch::nn` façade |
| `test_safetensors` | Save/load round-trips, sliced tensors, `LoadReport` |

### Models

| Target | Covers |
|---|---|
| `test_gpt2_primitives` | Attention, MLP, and block math |
| `test_gpt2_toy` | Training a small config end to end |
| `test_gpt2_huggingface` | Loading real GPT-2 weights and generating text |

`test_gpt2_huggingface` needs three files downloaded from [openai-community/gpt2](https://huggingface.co/openai-community/gpt2):

```bash
./build/test_gpt2_huggingface model.safetensors vocab.json merges.txt "The capital of France is"
```

### SIMD fused kernels

| Target | Extra flag |
|---|---|
| `test_avx2_f32_fk`, `test_avx2_d64_fk` | `-mfma` |
| `test_avx512_f32_fk`, `test_avx512_d64_fk` | `-mavx512f` |

> These are compiled **with** the ISA enabled and call the kernels directly, so they will crash with an illegal instruction on a CPU that lacks the feature. Check what your machine supports before running them (see [Backend](Backend.md)).

### Demos & benchmarks

| Target | What it does |
|---|---|
| `app` | AVX kernel timing benchmark |
| `trainmnist` / `testmnist` | Train and evaluate an MLP on MNIST |
| `convnet` / `ConvNet` / `InfConvNet` | Train and run inference with a CNN |
| `test_dashboard` | Live dashboard demo |

The MNIST targets expect the IDX files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, and the `t10k-*` pair) in the working directory.

---

## Installing

```bash
cmake --install build --prefix /your/prefix
```

This installs:

- every `.h` / `.hpp` under `src/` into `include/myproject/`, preserving the directory layout
- `libmyproject.a` into `lib/`
- a CMake package config into `lib/cmake/MyProject/`

### Consuming from another CMake project

```cmake
find_package(MyProject REQUIRED)
target_link_libraries(my_app PRIVATE MyProject::myproject)
```

Includes then resolve from the install root:

```cpp
#include "core.h"
#include "neuralnet.h"
```

---

## Umbrella Headers

Rather than including individual files, pull in a whole layer:

| Header | Brings in |
|---|---|
| `core.h` | Tensor, autograd, data, device, dispatch, ops, OpenMP kernels |
| `neuralnet.h` | Every layer, loss, metric, dataset, and serialization header |
| `cpu.h` | All AVX2/AVX-512 kernels, including fused ones |

`models/gpt2.h` and `models/gpt2_tokenizer.h` are included explicitly when needed.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `illegal instruction` in an `*_fk` test | The CPU lacks AVX2/AVX-512; those tests are compiled with the ISA enabled |
| Undefined references to `omp_*` | OpenMP runtime missing — install `libgomp` |
| MNIST target exits immediately | IDX data files not in the working directory |
| Dashboard page never loads | `python3` or `dashborad.py` missing; check `dashboard_log.txt` |
