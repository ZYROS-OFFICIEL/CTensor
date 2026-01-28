#pragma once
#include "core/tensor.h"
#include "core/autograd.h"
#include "neuralnet/module.h" 
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

// --- GRAD NODES ---

struct GradConv1d : GradFn {
    Tensor input, weight, bias;
    int stride, padding;
    GradConv1d(const Tensor& x, const Tensor& w, const Tensor& b, int s, int p)
        : input(x), weight(w), bias(b), stride(s), padding(p) { parents = {input, weight, bias}; }
    void backward(const Tensor& self) override;
};

struct GradConv2d: GradFn {
    Tensor input, weight, bias;
    int stride_h, stride_w, padding_h, padding_w;
    
    // Constructor matches the one called in conv.cpp
    GradConv2d(const Tensor& x, const Tensor& w, const Tensor& b, int sh, int sw, int ph, int pw)
        : input(x), weight(w), bias(b), stride_h(sh), stride_w(sw), padding_h(ph), padding_w(pw)
    { parents = {input, weight, bias}; }
    
    void backward(const Tensor& self) override;
};


struct GradConv3d: GradFn {
    Tensor input, weight, bias;
    int stride_d, stride_h, stride_w;
    int padding_d, padding_h, padding_w;

    // Constructor matches the one called in conv.cpp
    GradConv3d(const Tensor& x, const Tensor& w, const Tensor& b, int sd, int sh, int sw, int pd, int ph, int pw)
        : input(x), weight(w), bias(b), stride_d(sd), stride_h(sh), stride_w(sw), padding_d(pd), padding_h(ph), padding_w(pw)
    { parents = {input, weight, bias}; }
    
    void backward(const Tensor& self) override;
};