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
#include "tensor1.h"
#pragma once
#include "../tensor.h"

namespace nn {

class Conv1d {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight, bias;

    Conv1d(int in_c, int out_c, int k, int s=1, int p=0);
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

} 
class Conv2d {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight, bias;

    Conv2d(int in_c, int out_c, int k, int s=1, int p=0);
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

struct GradConv1d : GradFn {
    Tensor input, weight, bias;
    int stride, padding;

    GradConv1d(const Tensor& x, const Tensor& w, const Tensor& b, int s, int p)
        : input(x), weight(w), bias(b), stride(s), padding(p)
    {
        parents = {x, w, b};
    }

    void backward(const Tensor& self) override;
};
struct GradConv2d : GradFn {
    Tensor input, weight, bias;
    int stride, padding;

    GradConv2d(const Tensor& x, const Tensor& w, const Tensor& b, int s, int p)
        : input(x), weight(w), bias(b), stride(s), padding(p)
    {
        parents = {x, w, b};
    }

    void backward(const Tensor& self) override;
};