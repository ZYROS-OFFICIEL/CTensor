# DataLoader (Data Loader)

The `DataLoader` class is designed to efficiently feed batches of data from a dataset (like MNIST) into your neural network during training or evaluation.

It handles:
- Shuffling the dataset at the start of each epoch
- Slicing the dataset into smaller mini-batches
- On-the-fly type conversion

This allows you to store large datasets in memory using compact types (e.g., `UInt8`) while yielding batches as `Float32` for computation.

---

## Initialization

### Definition
Constructs a `DataLoader` wrapped around a dataset with configurable parameters.

### Parameters
- **dataset**: Reference to the dataset object (e.g., `MNISTData`)
- **bs**: Batch size
- **shuffle_data**: Whether to shuffle data each epoch
- **image_type**: Output data type for images (default: `Float32`)
- **label_type**: Output data type for labels (default: `Int32`)

### Usage
```cpp
// Initialize dataset
MNISTData mnist_train = load_mnist("path/to/data");

// Create DataLoader
DataLoader train_loader(mnist_train, 64, true, DType::Float32, DType::Int32);
````

### Returns

* `DataLoader`

---

## Reset Epoch (`reset`)

### Definition

Resets the internal pointer to the beginning of the dataset.

If shuffling is enabled, a new random permutation of indices is generated.

### Usage

```cpp
train_loader.reset();
```

### Returns

* `void`

---

## Fetch Next Batch (`next`)

### Definition

Retrieves the next mini-batch of data and labels.

* Uses optimized `memcpy` if data types match
* Otherwise performs safe element-wise casting
* Returns empty tensors when the dataset is exhausted

### Usage

```cpp
train_loader.reset();

while (true) {
    auto [batch_imgs, batch_labels] = train_loader.next();

    // End of epoch check
    if (!batch_imgs.impl) {
        break;
    }

    // Training step...
}
```

### Returns

* `std::pair<Tensor, Tensor>`
  (Image tensor and label tensor)

---

## Utility Methods

### Definition

Helper functions for dataset size and batching info.

### Usage

```cpp
size_t total_samples = train_loader.size();
size_t total_steps = train_loader.num_batches();
```

### Returns

* `size_t`

