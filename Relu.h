#pragma once
#include "tensor1.h"
#include "autograd.h"
#include <string>
#include <omp.h>


Tensor LeakyRelu(const Tensor& a_, double negative_slope = 0.01);

struct GradLeakyRelu : GradFn {
    Tensor a;
    double negative_slope;
    GradLeakyRelu(const Tensor& a_, double neg_slope) 
        : a(a_), negative_slope(neg_slope) {
        parents = {a};
    }
    void backward(const Tensor& self) override;
};



class PRelu {
public:
    Tensor weight;
    int num_parameters; 

    PRelu(int num_parameters = 1, double init = 0.25, DType dtype = DType::Float32);

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

struct GradPRelu : GradFn {
    Tensor input, weight;
    
    GradPRelu(const Tensor& input_, const Tensor& weight_) 
        : input(input_), weight(weight_) {
        parents = {input, weight}; 
    }
    void backward(const Tensor& self) override;
};