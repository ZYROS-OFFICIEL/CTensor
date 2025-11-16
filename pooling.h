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

//______________________Helpers________________________________
Tensor im2col_2d_pool(const Tensor& input,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w);
void col2im_2d_pool(const Tensor& grad_patches,
                    Tensor& grad_input,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w);
Tensor im2col_3d_pool(const Tensor& input,
                      int kernel_d, int kernel_h, int kernel_w,
                      int stride_d, int stride_h, int stride_w,
                      int pad_d, int pad_h, int pad_w);
void col2im_3d_pool(const Tensor& grad_patches,
                    Tensor& grad_input,
                    int kernel_d, int kernel_h, int kernel_w,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w);
                            




//----------------Pooling Classes---------------------------------------------
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

class MaxPool3d {
public:
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;

    MaxPool3d(int kd, int kh, int kw, int sd = 1, int sh = 1, int sw = 1, int pd = 0, int ph = 0, int pw = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
class AvgPool1d {
public:
    int kernel_size, stride, padding;

    AvgPool1d(int k, int s = 1, int p = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class AvgPool2d {
public:
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int padding_h, padding_w;

    AvgPool2d(int kh, int kw, int sh = 1, int sw = 1, int ph = 0, int pw = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
class AvgPool3d {
public:
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;

    AvgPool3d(int kd, int kh, int kw, int sd = 1, int sh = 1, int sw = 1, int pd = 0, int ph = 0, int pw = 0);

    // forward & call operator
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
//-------------------------------------------------------------------------------
struct GradAvgPool1d : public GradFn {
    Tensor input;
    int kernel_size;
    int stride;
    int padding;

    GradAvgPool1d(const Tensor& inp,
                  int k,
                  int s,
                  int p)
        : input(inp),
          kernel_size(k),
          stride(s),
          padding(p) {
        parents.push_back(input);
    }

    void backward(const Tensor& self) override;
};
struct GradAvgPool2d : public GradFn {
    Tensor input;
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int padding_h, padding_w;

    GradAvgPool2d(const Tensor& inp,
                  int kh, int kw,
                  int sh, int sw,
                  int ph, int pw)
        : input(inp),
          kernel_size_h(kh), kernel_size_w(kw),
          stride_h(sh), stride_w(sw),
          padding_h(ph), padding_w(pw) {
        parents.push_back(input);
    }

    void backward(const Tensor& self) override;
};