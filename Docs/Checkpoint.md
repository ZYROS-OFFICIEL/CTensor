
# Model Checkpointing (checkpoints)

The `checkpoints` namespace provides a simple, lightweight binary serialization system for saving and loading neural network weights to and from disk.

It uses a custom binary format safeguarded by a **"magic number"** to ensure file validity and strictly checks tensor shapes and data sizes during loading to prevent memory corruption or architecture mismatches.

---

## Save Weights (`save_weights`)

### Definition
Serializes a list of trainable parameters (`Tensor`s) and writes them to a binary file.  
It stores metadata (dimensions, shape, data size) followed by the raw contiguous memory of the tensor.

### Usage
```cpp
// Extract parameters from the model (e.g., using torch::nn::combine_params)
std::vector<Tensor*> params = model.parameters();

// Save to disk
checkpoints::save_weights(params, "my_model.bin");
````



---

## Load Weights (`load_weights`)

### Definition

Reads a previously saved binary checkpoint and copies the weights directly into the provided list of parameter `Tensor`s.

> **Note:** The architecture of the model providing the `params` vector must exactly match the architecture of the model that saved the weights.
> The function performs strict safety checks on the shape and byte size of every tensor before loading.

### Usage

```cpp
// Initialize the model first so the tensors are allocated with the correct shapes
MyModel model;
std::vector<Tensor*> params = model.parameters();

// Load weights from disk directly into the model's tensors
checkpoints::load_weights(params, "my_model.bin");
```


