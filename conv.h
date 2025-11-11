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



struct GradConv1d; // forward declare

class Conv1d {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight, bias;

    Conv1d(int in_c, int out_c, int k, int s = 1, int p = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class Conv2d {
public:
    int in_channels, out_channels, kernel_size_h, kernel_size_w;
    int stride_h, stride_w, padding_h, padding_w;
    Tensor weight, bias;

    Conv2d(int in_c, int out_c, int kh, int kw, int sh = 1, int sw = 1, int ph = 0, int pw = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// Grad node for Conv1d
struct GradConv1d : GradFn {
    Tensor input, weight, bias;
    int stride, padding;

    GradConv1d(const Tensor& x, const Tensor& w, const Tensor& b, int s, int p)
        : input(x), weight(w), bias(b), stride(s), padding(p) {
        parents = {input, weight, bias};
    }

    void backward(const Tensor& self) override;
};
struct GradConv2d: GradFn {
    Tensor input, weight, bias;
    int stride_h, stride_w, padding_h, padding_w;

    GradConv2d(const Tensor& x, const Tensor& w, const Tensor& b, int sh, int sw, int ph, int pw)
        : input(x), weight(w), bias(b), stride_h(sh), stride_w(sw), padding_h(ph), padding_w(pw)
    {
        parents = {input, weight, bias};
    }
    void backward(const Tensor& self) override;
};
