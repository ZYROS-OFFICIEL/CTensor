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
