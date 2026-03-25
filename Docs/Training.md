# Training & Optimization

This module provides all the necessary components to train a neural network, including datasets, data loaders, optimizers, learning rate schedulers, and standard training/evaluation loops.

---

## Data Loading (TensorDataset & SimpleDataLoader)

### Definition

**TensorDataset** pairs an input tensor (e.g., images) with a target tensor (e.g., labels) and can apply optional row-wise transformations.  
By default, if the input is `UInt8` (like raw pixels), it automatically normalizes them to `Float32` in the `[0, 1]` range.

**SimpleDataLoader** takes a dataset and automatically handles batching, shuffling, and multi-threaded data preparation during the training loop.

### Usage

```cpp
// Create a dataset from raw tensors
TensorDataset dataset(images_tensor, labels_tensor);

// Create a loader with a batch size of 64, shuffling enabled
SimpleDataLoader loader(dataset, 64, true);

// Iterate through batches manually
loader.reset();
while(loader.has_next()) {
    auto [batch_data, batch_labels] = loader.next();
    // ...
}
````

### Returns

`std::pair<Tensor, Tensor>`
(A tuple containing the batched Data and Target tensors)

---

## Model Modes

Neural network modules like Dropout or BatchNorm behave differently during training vs. inference.

### Set Model Mode (set_model_mode / set_train_mode)

#### Definition

Recursively sets the mode of a model or a list of layers to either training (`true`) or evaluation (`false`).

#### Usage

```cpp
// Set to training mode (enables dropout, tracks batch norm stats)
set_model_mode(my_model, true);

// Set to evaluation mode (disables dropout, uses accumulated batch norm stats)
set_model_mode(my_model, false);
```

#### Returns

`void`

---

## Weight Utilities

### Robust Initialization (robust_weight_init)

#### Definition

Initializes a list of parameter tensors with a robust uniform distribution scaled by a given factor.

#### Usage

```cpp
robust_weight_init(model.parameters(), 0.05f);
```

#### Returns

`void` (Modifies weights in-place)

---

### Corruption Check (check_weights_corrupted)

#### Definition

Quickly scans a sample of a model's weights to detect NaNs, Infs, or explosively large values (> 50.0). Useful for debugging exploding gradients.

#### Usage

```cpp
bool is_corrupted = check_weights_corrupted(model.parameters());
```

#### Returns

`bool`

---

## Optimizers (torch::optim)

### Definition

Optimizers dictate how a model's weights are updated based on the gradients computed during the `.backward()` pass.

All optimizers inherit from a base `Optimizer` class and implement:

* `.zero_grad()` to clear buffers
* `.step()` to apply updates

### Available Optimizers

* SGD: Standard Stochastic Gradient Descent
* Adam: Adaptive Moment Estimation (momentum + RMS scaling)
* AdamW: Adam with decoupled weight decay (L2 regularization)
* Adamax: Adam based on the infinity norm
* NAdam: Adam with Nesterov momentum
* RAdam: Rectified Adam (improves variance in early steps)
* RMSprop: Root Mean Square Propagation
* Adagrad: Adaptive Gradient algorithm
* Lion: Evolved Sign Momentum optimizer (memory efficient)

### Usage

```cpp
// Initialize Optimizer (e.g., AdamW)
torch::optim::AdamW optimizer(model.parameters(), 0.001); // lr = 0.001

// Standard training step
optimizer.zero_grad();      // 1. Clear old gradients
Tensor loss = ...;          // 2. Compute loss
backward(loss);             // 3. Compute new gradients
optimizer.step();           // 4. Update weights
```

### Returns

`void`

---

## Learning Rate Schedulers

### StepLR

#### Definition

Decays the learning rate of an optimizer by a multiplicative factor (`gamma`) every `step_size` epochs.

#### Usage

```cpp
// Multiply LR by 0.1 every 10 epochs
StepLR scheduler(optimizer, 10, 0.1);

for(int epoch = 0; epoch < 100; ++epoch) {
    train_epoch(...);
    scheduler.step(); // Call at the end of the epoch
}
```

#### Returns

`void`

---

## Training & Evaluation Loops

Ready-to-use template functions that abstract away the boilerplate of iterating over a dataloader.

---

### Train Epoch (train_epoch)

#### Definition

Iterates through one entire epoch of a dataloader:

* Computes forward pass
* Applies Cross-Entropy loss
* Executes backward pass
* Steps the optimizer

Automatically logs progress to the console.

#### Usage

```cpp
// Train for one epoch, logging progress every 100 batches
train_epoch(model, dataloader, optimizer, current_epoch, 100);
```

#### Returns

`void`

---

### Evaluate (evaluate)

#### Definition

Iterates through a dataloader in evaluation mode:

* Computes Cross-Entropy loss
* Computes standard accuracy

Automatically logs metrics to the web dashboard server (`log_metrics`).

#### Usage

```cpp
set_model_mode(model, false); // Important: Set to eval mode first!
double accuracy = evaluate(model, test_dataloader);
```

#### Returns

`double`
(The overall accuracy percentage on the dataset)

