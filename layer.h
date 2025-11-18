#pragma once
#include "tensor1.h"
#include "ops1.h"
#include "autograd.h"
#include <string>

// --- Linear (Dense/Fully Connected) Layer ---
// Applies a linear transformation to the incoming data: y = xA^T + b
class Linear {
public:
    int in_features;
    int out_features;
    Tensor weight; // Shape: [out_features, in_features]
    Tensor bias;   // Shape: [out_features]

    // Constructor
    Linear(int in_feat, int out_feat, bool bias = true);

    // Forward pass
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- Flatten Layer ---
// Flattens a contiguous range of dims into a tensor.
// Typically used to flatten all non-batch dimensions: [B, C, H, W] -> [B, C*H*W]
class Flatten {
public:
    int start_dim;
    int end_dim; // -1 means last dimension

    Flatten(int start_dim = 1, int end_dim = -1) 
        : start_dim(start_dim), end_dim(end_dim) {}

    Tensor forward(const Tensor& input) const;
    Tensor operator()(const Tensor& input) const { return forward(input); }
};