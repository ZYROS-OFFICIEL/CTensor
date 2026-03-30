# Dropout

**Dropout** is a highly effective regularization technique used to prevent neural networks from overfitting. It works by randomly zeroing out elements of the input tensor with a specified probability `p` during training. This prevents neurons from co-adapting too much and forces the network to learn robust, redundant representations.

This specific implementation utilizes **Inverted Dropout**. During training, the non-zeroed elements are automatically scaled up by a factor of `1.0 / (1.0 - p)`. This ensures that the expected value (and overall magnitude) of the activations remains constant, meaning no manual scaling is required during the inference/evaluation phase.

---

## Dropout Layer (`Dropout`)

### Definition

A layer class (inheriting from `Module`) that applies inverted dropout to the input tensor.

Crucially, this layer respects the internal training state of the module:

- When the model is in **training mode** (default), random masking is applied.
- When switched to **evaluation mode** (using `.eval()`), the layer acts as an identity function, returning the input unaltered.

---

### Parameters

- `p`: The probability of an element being zeroed out.  
  - Range: `[0.0, 1.0)`  
  - Default: `0.5`

---

### Usage

```cpp
// Initialize a Dropout layer with a 30% probability of dropping elements
Dropout drop(0.3);

// --- Training Phase ---
// Explicitly set to training mode (or use set_model_mode(model, true))
drop.train();
Tensor train_output = drop(input_tensor); // Random masking applied

// --- Evaluation Phase ---
// Explicitly set to eval mode (or use set_model_mode(model, false))
drop.eval();
Tensor test_output = drop(input_tensor); // Returns input_tensor identically
````

---

### Returns

* `Tensor` (same shape as input)

