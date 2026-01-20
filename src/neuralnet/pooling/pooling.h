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

// --- MaxPool3d ---
class MaxPool3d : public Module {
public:
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;

    // FIX: Default strides to -1
    MaxPool3d(int kd, int kh, int kw, int sd = -1, int sh = -1, int sw = -1, int pd = 0, int ph = 0, int pw = 0);

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- AvgPool Classes (Apply same fix) ---

class AvgPool1d : public Module {
public:
    int kernel_size, stride, padding;
    AvgPool1d(int k, int s = -1, int p = 0); // FIX
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class AvgPool2d : public Module {
public:
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int padding_h, padding_w;
    AvgPool2d(int kh, int kw, int sh = -1, int sw = -1, int ph = 0, int pw = 0); // FIX
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

class AvgPool3d : public Module {
public:
    int kernel_size_d, kernel_size_h, kernel_size_w;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;
    AvgPool3d(int kd, int kh, int kw, int sd = -1, int sh = -1, int sw = -1, int pd = 0, int ph = 0, int pw = 0); // FIX
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// ... Grad Nodes ...
struct GradAvgPool1d : public GradFn {
    Tensor input;
    int kernel_size, stride, padding;
    GradAvgPool1d(const Tensor& inp, int k, int s, int p) 
        : input(inp), kernel_size(k), stride(s), padding(p) { parents.push_back(input); }
    void backward(const Tensor& self) override;
};
