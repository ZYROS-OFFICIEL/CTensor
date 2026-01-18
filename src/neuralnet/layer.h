#pragma once
#include "tensor1.h"
#include "opsmp.h"
#include "autograd.h"
#include "module.h" // Inherit from Module
#include <string>
#include <vector>

// --- Linear (Dense/Fully Connected) Layer ---
// Applies a linear transformation to the incoming data: y = xA^T + b
class Linear : public Module {
public:
    int in_features;
    int out_features;
    DType dtype;
    Tensor weight; // Shape: [out_features, in_features]
    Tensor bias;   // Shape: [out_features]

    // Constructor
    Linear(int in_feat, int out_feat, bool bias = true, DType dt = DType::Float32);

    // Implement parameters() so the Optimizer can train this layer
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> params = {&weight};
        if (bias.impl) params.push_back(&bias);
        return params;
    }

    // Forward pass
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
