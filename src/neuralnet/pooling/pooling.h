#pragma once
#include "tensor.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include "module.h"

// ... helpers ...

// --- MaxPool1d ---
class MaxPool1d : public Module {
public:
    int kernel_size, stride, padding;

    MaxPool1d(int k, int s = -1, int p = 0);

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- MaxPool2d ---
class MaxPool2d : public Module {
public:
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int padding_h, padding_w;

    // FIX: Default strides to -1
    MaxPool2d(int kh, int kw, int sh = -1, int sw = -1, int ph = 0, int pw = 0);

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
