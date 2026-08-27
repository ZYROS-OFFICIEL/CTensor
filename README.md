# CTensor

A tensor and deep learning library written from scratch in C++20 — no BLAS, no ONNX runtime, no Python. Autograd, neural network layers, optimizers, hand-written AVX2/AVX-512 kernels, and a complete GPT-2 that loads the official Hugging Face weights and generates text.

If you know PyTorch, you already know the API.

```cpp
#include "core.h"
#include "neuralnet.h"

using namespace torch;

class MLPNet : public nn::Module {
public:
    nn::Flatten flat;
    nn::Linear fc1{784, 128};
    nn::Linear fc2{128, 10};

    Tensor forward(const Tensor& x) {
        return fc2(nn::functional::relu(fc1(flat(x))));
    }
    Tensor operator()(const Tensor& x) { return forward(x); }

    std::vector<Tensor*> parameters() override {
        return nn::combine_params(fc1, fc2);
    }
};

int main() {
    MLPNet model;
    auto params = model.parameters();
    nn::init::kaiming_uniform_(params);

    optim::AdamW optimizer(params, 1e-3);
    auto criterion = nn::CrossEntropyLoss();

    auto dataset = vision::datasets::MNIST("train-images.idx3-ubyte",
                                           "train-labels.idx1-ubyte");
    DataLoader loader(dataset, 64, true);

    model.train();
    for (int epoch = 1; epoch <= 5; ++epoch) {
        for (auto& batch : loader) {
            optimizer.zero_grad();
            Tensor loss = criterion(model(batch.data), batch.target);
            loss.backward();
            nn::utils::clip_grad_norm_(params, 1.0);
            optimizer.step();
        }
    }

    checkpoints::save_weights(params, "mnist.bin");
}
```

---

## Features

**Tensor engine** — N-dimensional tensors with strides, views, slicing, broadcasting, and 13 dtypes from `Bool` to `Double64`.

**Autograd** — reverse-mode automatic differentiation. Every operation records its own gradient node; `loss.backward()` walks the graph.

**Layers** — `Linear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, pooling, `Dropout`, `BatchNorm`, `LayerNorm`, `Embedding`, `MultiheadAttention`, `Upsample`.

**Training** — nine optimizers (SGD, Adam, AdamW, Adamax, NAdam, RAdam, RMSprop, Adagrad, Lion), LR schedulers, gradient clipping, ready-made train/eval loops, and a live browser dashboard.

**Performance** — hand-written AVX2 and AVX-512 kernels for `f32` and `f64`, selected at runtime by CPU feature detection, with an OpenMP-threaded fallback that runs anywhere. Plus a catalogue of fused kernels (`fma`, `swiglu`, `layer_norm`, `bias_add_gelu`, …) that compute multi-step expressions in a single pass over memory.

**Interoperability** — full [safetensors](https://github.com/huggingface/safetensors) read/write with a dependency-free JSON parser, so weights move between CTensor and PyTorch without a conversion script.

**GPT-2** — the architecture, a faithful byte-level BPE tokenizer, and a loader that handles the `Conv1D` transposes and fused-QKV splits in the official checkpoint. Load `model.safetensors`, encode a prompt, generate.

---

## Build

Requires CMake ≥ 3.26, a C++20 compiler (GCC or Clang), and OpenMP on x86-64.

```bash
git clone https://github.com/ZYROS-OFFICIEL/CTensor
cd CTensor
cmake -B build
cmake --build build -j$(nproc)
```

This produces `build/libmyproject.a` and the test and demo executables. Full build, test, and install instructions are in [Docs/Build.md](Docs/Build.md).

### Use it from your own project

```cmake
find_package(MyProject REQUIRED)
target_link_libraries(my_app PRIVATE MyProject::myproject)
```

---

## Run GPT-2

Download the three files from [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) — `model.safetensors`, `vocab.json`, `merges.txt` — then:

```bash
cmake --build build --target test_gpt2_huggingface
./build/test_gpt2_huggingface model.safetensors vocab.json merges.txt "The capital of France is"
```

---

## Documentation

Start with **[Getting Started](Docs/index.md)**, which also indexes the full API reference.

| | |
|---|---|
| **Core** | [Tensors](Docs/Tensor.md) · [Operations](Docs/Ops.md) · [Autograd](Docs/Autograd.md) · [Devices](Docs/Device.md) · [Backend & Dispatch](Docs/Backend.md) |
| **Models** | [Modules](Docs/Module.md) · [nn](Docs/nn.md) · [Layers](Docs/Layer.md) · [Conv](Docs/Conv.md) · [ConvTranspose](Docs/ConvTranspose.md) · [Pooling](Docs/Pooling.md) · [ReLU](Docs/relu.md) · [Dropout](Docs/Dropout.md) · [BatchNorm](Docs/BatchNorm.md) · [Norms](Docs/Norm.md) · [Embedding](Docs/Embedding.md) · [Attention](Docs/Attention.md) · [Functional](Docs/Functional.md) · [Weights](Docs/Weights.md) |
| **Training** | [Data](Docs/Data.md) · [DataLoader](Docs/Dataloader.md) · [Training](Docs/Training.md) · [Loss](Docs/Loss.md) · [Metrics](Docs/Metrics.md) · [Dashboard](Docs/Dashboard.md) |
| **I/O** | [Checkpoints](Docs/Checkpoint.md) · [Serialization](Docs/Serialization.md) |
| **Models** | [GPT-2](Docs/GPT2.md) |
| **Project** | [Build & Test](Docs/Build.md) |
