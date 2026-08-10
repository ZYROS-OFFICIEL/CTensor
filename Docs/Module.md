# Modules & Containers (`Module`, `Sequential`)

Every layer in CTensor derives from the base `Module` class. A module owns its learnable `Tensor`s, exposes them to optimizers and serialization, and tracks whether it is currently in training or evaluation mode.

---

## Base Class (`Module`)

### Definition

`Module` is the abstract base every layer inherits from. It provides:

- **`training`**: a `bool` flag, `true` by default
- **`train()` / `eval()`**: virtual mode switches. Layers whose behaviour depends on the mode (`Dropout`, `BatchNorm`) read this flag; containers override these to recurse into their children
- **`parameters()`**: returns a flat `std::vector<Tensor*>` of every learnable tensor. This is what you hand to an optimizer
- **`named_parameters(prefix)`**: returns a `NamedParams` list — `std::vector<std::pair<std::string, Tensor*>>` — pairing each parameter with a dotted path such as `"fc1.weight"`. This is what checkpoint formats key on

A custom layer must implement `forward()`, override `parameters()`, and — if you intend to save it by name — override `named_parameters()`.

### Usage

```cpp
class MLP : public Module {
public:
    Linear fc1{784, 128};
    Linear fc2{128, 10};

    Tensor forward(const Tensor& x) { return fc2(relu(fc1(x))); }
    Tensor operator()(const Tensor& x) { return forward(x); }

    std::vector<Tensor*> parameters() override {
        return torch::nn::combine_params(fc1, fc2);
    }

    NamedParams named_parameters(const std::string& prefix = "") override {
        NamedParams p;
        collect_named(p, prefix, "fc1", fc1);   // -> "fc1.weight", "fc1.bias"
        collect_named(p, prefix, "fc2", fc2);   // -> "fc2.weight", "fc2.bias"
        return p;
    }
};
```

### Returns

| Method | Return Type |
|---|---|
| `parameters()` | `std::vector<Tensor*>` |
| `named_parameters(prefix)` | `Module::NamedParams` |
| `train()` / `eval()` | `void` |

---

## Named Parameter Helper (`collect_named`)

### Definition

Free function that appends a submodule's named parameters to a parent's list, prefixing each entry with `parent_prefix + name + "."`.

```cpp
void collect_named(Module::NamedParams& out,
                   const std::string& prefix,
                   const std::string& name,
                   Module& m);
```

Calling it recursively is what produces PyTorch-style flat paths like `h.0.attn.q_proj.weight`, which lets a CTensor model load a checkpoint written by another framework.

### Usage

```cpp
NamedParams p;
collect_named(p, prefix, "attn", attn);
collect_named(p, prefix, "mlp",  mlp);
```

### Returns

`void` (appends to `out`)

---

## Sequential Container (`Sequential`)

### Definition

Holds an ordered list of `std::shared_ptr<Module>` children and forwards the module protocol to all of them:

- `train()` / `eval()` recurse into every child
- `parameters()` concatenates every child's parameters
- `named_parameters()` names children by their **index** (`"0.weight"`, `"1.weight"`, …)

> **Note:** `Sequential` is a parameter/mode container. It does not define a `forward()` chain — call the child modules yourself in your model's `forward`.

### Usage

```cpp
Sequential features;
features.add(std::make_shared<Conv2d>(1, 16, 3));
features.add(std::make_shared<ReLU>());
features.add(std::make_shared<MaxPool2d>(2, 2));

features.eval();                          // recurses into all three
auto params = features.parameters();      // all learnable tensors, in order
```

### Returns

* `std::vector<Tensor*>` from `parameters()`
* `Module::NamedParams` from `named_parameters()`

---

## Training vs. Evaluation Mode

Modules with stochastic or statistics-tracking behaviour branch on `training`:

| Layer | `train()` | `eval()` |
|---|---|---|
| `Dropout` | zeroes elements with probability `p`, rescales by `1/(1-p)` | identity |
| `BatchNorm` | normalizes with batch statistics, updates running stats | normalizes with running statistics |

Composite models must propagate the switch. Layers that own submodules override `train()`/`eval()` and forward the call, as `GPT2Block` and `GPT2Model` do:

```cpp
void train() override { Module::train(); attn.train(); mlp.train(); }
void eval()  override { Module::eval();  attn.eval();  mlp.eval();  }
```

`set_model_mode(model, bool)` (see [Training](Training.md)) does the same job for models and plain layer lists.

### Returns

`void`
