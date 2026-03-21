# Loss Functions (Loss)

The `Loss` class provides a collection of static methods for computing the error (loss) between a model's predictions and the actual target values. These functions automatically build the computation graph for backpropagation.

Most loss functions support a `reduction` parameter, which determines how the batch of losses is aggregated. Typically, this can be "mean" (default) or "sum".

---

## Regression Losses

These functions are typically used for continuous numerical prediction tasks.

### Mean Squared Error (MSE)

**Definition**  
Computes the mean squared error (squared L2 norm) between each element in the prediction and target tensors.

**Usage**  
```cpp
Tensor loss = Loss::MSE(predictions, targets);
````

**Returns:**
Tensor (Scalar)

---

### Mean Absolute Error (MAE)

**Definition**
Computes the mean absolute error (L1 norm) between predictions and targets. Less sensitive to outliers than MSE.

**Usage**

```cpp
Tensor loss = Loss::MAE(predictions, targets, "mean");
```

**Returns:**
Tensor (Scalar)

---

### Huber Loss & Smooth L1 Loss (HuberLoss, SmoothL1Loss)

**Definition**
Uses a squared term if the absolute error falls below a delta threshold and an absolute term otherwise. It combines the best properties of L1 and L2 loss. SmoothL1Loss is simply a convenient wrapper around HuberLoss with delta = 1.0.

**Usage**

```cpp
Tensor huber = Loss::HuberLoss(predictions, targets, "mean", 1.5); // Custom delta
Tensor smooth_l1 = Loss::SmoothL1Loss(predictions, targets, "mean"); // Delta = 1.0
```

**Returns:**
Tensor (Scalar)

---

### Log-Cosh Loss (LogCosh)

**Definition**
Computes the logarithm of the hyperbolic cosine of the prediction error. It is approximately `(x**2)/2` for small x and `abs(x) - log(2)` for large x.

**Usage**

```cpp
Tensor loss = Loss::LogCosh(predictions, targets);
```

**Returns:**
Tensor (Scalar)

---

## Classification Losses

These functions are designed for categorical or probability-based predictions.

### Cross Entropy (CrossEntropy)

**Definition**
Computes the cross-entropy loss between input logits and target labels. It automatically computes the Log-Softmax internally, so the predictions should be raw, unnormalized logits.
**Note:** The target tensor MUST contain integer class indices of shape [Batch, 1], not one-hot vectors.

**Usage**

```cpp
Tensor logits = model(inputs); // e.g., Shape [32, 10] for 10 classes
Tensor labels = ...;           // Shape [32, 1] containing values 0-9
Tensor loss = Loss::CrossEntropy(logits, labels);
```

**Returns:**
Tensor (Scalar)

---

### Negative Log Likelihood (NLLLoss)

**Definition**
The negative log-likelihood loss. It is similar to Cross Entropy, but expects the inputs to already be log-probabilities (e.g., passed through a LogSoftmax layer). Targets must be integer indices.

**Usage**

```cpp
Tensor loss = Loss::NLLLoss(log_probs, labels);
```

**Returns:**
Tensor (Scalar)

---

### Binary Cross Entropy (BCE)

**Definition**
Computes the binary cross-entropy loss.
**Note:** The inputs must represent probabilities strictly in the range [0, 1] (e.g., passed through a Sigmoid layer).

**Usage**

```cpp
Tensor probs = torch::nn::functional::sigmoid(logits);
Tensor loss = Loss::BCE(probs, targets);
```

**Returns:**
Tensor (Scalar)

---

### Kullback-Leibler Divergence (KLDiv)

**Definition**
Computes the KL divergence loss, a measure of how one probability distribution diverges from a second, expected probability distribution.

**Usage**

```cpp
Tensor loss = Loss::KLDiv(predictions, targets);
```

**Returns:**
Tensor (Scalar)

---

### Hinge Loss (HingeLoss)

**Definition**
Commonly used for "maximum-margin" classification, most notably for support vector machines (SVMs).

**Usage**

```cpp
Tensor loss = Loss::HingeLoss(predictions, targets);
```

**Returns:**
Tensor (Scalar)

---

## Ranking Losses

Functions used to predict the relative distances or rankings between inputs.

### Margin Ranking Loss (MarginRankingLoss)

**Definition**
Creates a criterion that measures the loss given inputs x1, x2, and a label tensor y (containing 1 or -1). It evaluates whether x1 is ranked higher than x2 by at least the given margin.

**Usage**

```cpp
// Target contains 1 (if input1 should be larger) or -1 (if input2 should be larger)
Tensor loss = Loss::MarginRankingLoss(input1, input2, target, 0.5, "mean");
```

**Returns:**
Tensor (Scalar)

---
