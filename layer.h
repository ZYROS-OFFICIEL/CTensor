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
