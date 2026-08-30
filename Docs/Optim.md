# Optimizers in CTensor

Optimizers are essential components in the training pipeline of a neural network. They define the specific strategy used to update the model's internal parameters (weights and biases) based on the computed gradients, aiming to minimize the target loss function efficiently.

## Supported Algorithms

This library provides implementations of several widely-used optimization algorithms, optimized for efficient tensor operations:

* **SGD (Stochastic Gradient Descent):** The foundational optimizer that updates parameters by taking a step in the direction of the negative gradient.
* **Momentum:** An extension of standard SGD that accumulates a velocity vector in directions of persistent reduction in the objective across iterations, significantly reducing oscillation.
* **Adam (Adaptive Moment Estimation):** A highly popular optimizer that computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
* **RMSprop:** An algorithm that adapts the learning rate for each parameter by dividing the gradient by a running average of its recent magnitude, performing well in non-stationary settings.

## Hyperparameters

When initializing an optimizer, you must specify its configuration parameters. The necessary fields depend on the chosen algorithm:

* **Learning Rate (`lr`):** The primary scalar value controlling the step size during the parameter update (required for all optimizers).
* **Momentum Factor:** Used in Momentum-based SGD to determine the contribution of the previous gradient step to the current update.
* **Beta 1 & Beta 2:** The exponential decay rates for the first and second moment estimates, strictly used for Adam.
* **Epsilon:** A very small constant (e.g., `1e-8`) added to the denominator in Adam and RMSprop to prevent division by zero errors.

## Usage Example

Here is a conceptual example of how to configure and apply an optimizer within your standard C training loop using `train_utils.h`:

```c
#include "train_utils.h"
/* Include other necessary CTensor headers */

int main() {
    /* 1. Initialize your model/network */
    /* NeuralNetwork* nn = ... */

    /* 2. Initialize Adam with standard hyperparameters (lr, beta1, beta2, epsilon) */
    Optimizer* opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);

    /* 3. Main training loop */
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        /* a. Perform the forward pass */
        /* b. Compute the loss */
        /* c. Perform the backward pass to calculate gradients */
        
        /* d. Update the neural network's parameters */
        optimizer_step(opt, nn);
        
        /* e. Zero the gradients for the next iteration */
        optimizer_zero_grad(opt, nn);
    }

    /* 4. Clean up memory */
    optimizer_free(opt);
    
    return 0;
}
```