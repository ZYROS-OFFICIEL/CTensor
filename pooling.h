#include "tensor1.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "ops1.h"
#include <string>
#pragma once

class MaxPool1d {
public:
    int kernel_size, stride, padding;

    MaxPool1d(int k, int s = 1, int p = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
class MaxPool2d {
public:
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int padding_h, padding_w;

    MaxPool2d(int kh, int kw, int sh = 1, int sw = 1, int ph = 0, int pw = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
