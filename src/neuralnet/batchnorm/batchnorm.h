#pragma once
#include "tensor.h"
#include "autograd.h"

// --- BatchNorm1d / BatchNorm2d ---
// Applies Batch Normalization over a 2D or 4D input.
// 2D Input: [N, C] -> Normalizes over N for each C.
// 4D Input: [N, C, H, W] -> Normalizes over (N, H, W) for each C.

class BatchNorm {
public:
    int num_features;
    double eps;
    double momentum;
    bool training; // Flag to switch between training and inference modes

    // Learnable parameters
    Tensor gamma; // Scale
    Tensor beta;  // Shift

    // Running statistics (not learned via gradient descent)
    Tensor running_mean;
    Tensor running_var;

    BatchNorm(int num_features, double eps = 1e-5, double momentum = 0.1);

    // Switch modes
    void train() { training = true; }
    void eval() { training = false; }

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};