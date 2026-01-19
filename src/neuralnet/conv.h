#pragma once
#include "tensor.h"
#include "autograd.h"
#include "module.h" // Inherit from Module
#include <vector>

// --- Conv1d ---
class Conv1d : public Module {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight, bias;

    // Added DType to match your cpp implementation
    Conv1d(int in_c, int out_c, int k, int s = 1, int p = 0, DType dt = DType::Float32);

    std::vector<Tensor*> parameters() override { return {&weight, &bias}; }
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
