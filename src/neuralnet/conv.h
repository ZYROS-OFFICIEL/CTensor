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

// --- Conv2d ---
class Conv2d : public Module {
public:
    int in_channels, out_channels, kernel_size_h, kernel_size_w;
    int stride_h, stride_w, padding_h, padding_w;
    Tensor weight, bias;

    // Added DType
    Conv2d(int in_c, int out_c, int kh, int kw, int sh = 1, int sw = 1, int ph = 0, int pw = 0, DType dt = DType::Float32);

    std::vector<Tensor*> parameters() override { return {&weight, &bias}; }
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- Conv3d ---
class Conv3d : public Module {
public:
    int in_channels, out_channels;
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;
    Tensor weight, bias;

    // Added DType
    Conv3d(int in_c, int out_c, int kd, int kh, int kw, int sd = 1, int sh = 1, int sw = 1, int pd = 0, int ph = 0, int pw = 0, DType dt = DType::Float32);

    std::vector<Tensor*> parameters() override { return {&weight, &bias}; }
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
