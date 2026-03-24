
# Metrics (`torch::metrics`)

The `torch::metrics` namespace provides a collection of standard evaluation functions to measure the performance of your machine learning models.

Unlike loss functions, these metrics are typically not differentiable and do not build a computation graph. They return standard C++ types (`float` or `size_t`) instead of Tensors, making them ideal for logging, evaluation, and monitoring training progress.

---

## Multiclass Classification

These metrics are designed for evaluating models predicting across mutually exclusive classes.

### Accuracy Count (`accuracy`)

**Definition**

Calculates the raw number of correct predictions in a batch. It applies an `argmax` operation over the class dimension (dimension 1) of the predictions to find the predicted class index and compares it against the target labels.

This is highly useful for accumulating the total number of correct predictions across multiple batches during an epoch.

**Usage**

```cpp
// predictions: [Batch, NumClasses] (raw logits)
// targets: [Batch] or [Batch, 1] (class indices)
size_t correct = torch::metrics::accuracy(predictions, targets);
````

**Returns:**
`size_t` — The integer count of correct predictions

---

### Accuracy Score (`accuracy_score`)

**Definition**

Computes the standard accuracy metric as a ratio (between `0.0` and `1.0`) of correct predictions to the total number of predictions in the batch.

**Usage**

```cpp
float acc = torch::metrics::accuracy_score(predictions, targets);
```

**Returns:**
`float` — Value in `[0.0, 1.0]`

---

## Binary Classification

These metrics evaluate models predicting a single probability output (usually between `0.0` and `1.0`, e.g., after a Sigmoid activation).

All binary metrics accept an optional threshold parameter (defaults to `0.5`).

### Binary Accuracy (`binary_accuracy`)

**Definition**

Measures the percentage of correct predictions where both the target and the thresholded prediction match.

**Usage**

```cpp
float b_acc = torch::metrics::binary_accuracy(predictions, targets, 0.5f);
```

**Returns:**
`float` — Value in `[0.0, 1.0]`

---

### Precision Score (`precision_score`)

**Definition**

Computes the Precision metric:

```
True Positives / (True Positives + False Positives)
```

It measures how many of the positively predicted instances were actually positive.

**Usage**

```cpp
float precision = torch::metrics::precision_score(predictions, targets);
```

**Returns:**
`float` — Value in `[0.0, 1.0]`

---

### Recall Score (`recall_score`)

**Definition**

Computes the Recall metric:

```
True Positives / (True Positives + False Negatives)
```

It measures how many of the actual positive instances the model correctly identified.

**Usage**

```cpp
float recall = torch::metrics::recall_score(predictions, targets);
```

**Returns:**
`float` — Value in `[0.0, 1.0]`

---

### F1 Score (`f1_score`)

**Definition**

Computes the harmonic mean of Precision and Recall. This is an excellent holistic metric for evaluating binary classifiers, especially on imbalanced datasets.

**Usage**

```cpp
float f1 = torch::metrics::f1_score(predictions, targets);
```

**Returns:**
`float` — Value in `[0.0, 1.0]`

---

## Regression Metrics

These metrics evaluate the predictive performance of models outputting continuous numerical values.

### Mean Squared Error (`mean_squared_error`)

**Definition**

Computes the mean of the squared differences between predictions and targets.

**Usage**

```cpp
float mse = torch::metrics::mean_squared_error(predictions, targets);
```

**Returns:**
`float`

---

### Mean Absolute Error (`mean_absolute_error`)

**Definition**

Computes the mean of the absolute differences between predictions and targets.

**Usage**

```cpp
float mae = torch::metrics::mean_absolute_error(predictions, targets);
```

**Returns:**
`float`

---

### R-Squared Score (`r2_score`)

**Definition**

Computes the Coefficient of Determination ($R^2$).

It provides a measure of how well the observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.

* A perfect model yields an $R^2$ of `1.0`
* A model that always predicts the mean yields `0.0`
* It can be negative if the model performs worse than this baseline

**Usage**

```cpp
float r2 = torch::metrics::r2_score(predictions, targets);
```

**Returns:**
`float` (usually ≤ `1.0`)

