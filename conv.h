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


//----------------Helpers---------------------------------------------
Tensor im2col_2d(const Tensor& input,
                 int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w);

void col2im_2d(const Tensor& grad_patches,
               Tensor& grad_input,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w);
Tensor im2col_3d(const Tensor& input,
                 int kernel_d, int kernel_h, int kernel_w,
                 int stride_d, int stride_h, int stride_w,
                 int pad_d, int pad_h, int pad_w);

void col2im_3d(const Tensor& grad_patches,
               Tensor& grad_input,
               int kernel_d, int kernel_h, int kernel_w,
               int stride_d, int stride_h, int stride_w,
               int pad_d, int pad_h, int pad_w);


class Conv1d {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weight, bias;

    Conv1d(int in_c, int out_c, int k, int s = 1, int p = 0,DType dt = DType::Double64);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class Conv2d {
public:
    int in_channels, out_channels, kernel_size_h, kernel_size_w;
    int stride_h, stride_w, padding_h, padding_w;
    Tensor weight, bias;

    Conv2d(int in_c, int out_c, int kh, int kw, int sh = 1, int sw = 1, int ph = 0, int pw = 0,DType dt = DType::Double64);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class Conv3d {
public:
    int in_channels, out_channels;
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;
    Tensor weight, bias;

    Conv3d(int in_c, int out_c, int kd, int kh, int kw, int sd = 1, int sh = 1, int sw = 1, int pd = 0, int ph = 0, int pw = 0,DType dt = DType::Double64);

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
// --- NEW: Grad node for matmul-based conv2d ---
struct GradConv2dMatmul : GradFn {
    Tensor input, weight, bias, input_patches;
    int stride_h, stride_w, pad_h, pad_w;
    int kernel_h, kernel_w;
    GradConv2dMatmul(const Tensor& x, const Tensor& w, const Tensor& b,
                     const Tensor& patches,
                     int sh, int sw, int ph, int pw,
                     int kh, int kw)
        : input(x), weight(w), bias(b), input_patches(patches),
          stride_h(sh), stride_w(sw), pad_h(ph), pad_w(pw),
          kernel_h(kh), kernel_w(kw)
    {
        parents = {input, weight, bias};
    }
    void backward(const Tensor& self) override;
};

// --- NEW: Grad node for matmul-based conv3d ---
struct GradConv3dMatmul : GradFn {
    Tensor input, weight, bias, input_patches;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    int kernel_d, kernel_h, kernel_w;
    GradConv3dMatmul(const Tensor& x, const Tensor& w, const Tensor& b,
                     const Tensor& patches,
                     int sd, int sh, int sw, int pd, int ph, int pw,
                     int kd, int kh, int kw)
        : input(x), weight(w), bias(b), input_patches(patches),
          stride_d(sd), stride_h(sh), stride_w(sw),
          pad_d(pd), pad_h(ph), pad_w(pw),
          kernel_d(kd), kernel_h(kh), kernel_w(kw)
    {
        parents = {input, weight, bias};
    }
    void backward(const Tensor& self) override;
};
