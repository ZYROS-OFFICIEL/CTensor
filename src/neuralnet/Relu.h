#pragma once
#include "tensor.h"
#include "autograd.h"
#include "module.h" // Inherit from Module

// --- Standard ReLU (Layer) ---
// Wrapper around the functional Relu_mp for Module compatibility
class Relu : public Module {
public:
    Relu() {}
    // No parameters

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- Leaky ReLU (Function/Op) ---
// Stateless: slope is fixed.
Tensor LeakyRelu(const Tensor& input, double negative_slope = 0.01);

struct GradLeakyRelu : GradFn {
    Tensor input;
    double negative_slope;
    GradLeakyRelu(const Tensor& inp, double slope) 
        : input(inp), negative_slope(slope) {
        parents = {input};
    }
    void backward(const Tensor& self) override;
};


// --- PReLU (Layer/Class) ---
// Stateful: 'weight' (alpha) is learnable.
class PRelu : public Module {
public:
    Tensor weight; // The learnable slope(s)
    int num_parameters; // 1 (shared) or C (per-channel)

    // init: initial value for alpha (usually 0.25)
    PRelu(int num_parameters = 1, double init = 0.25);

    // Implement parameters() so Optimizer finds the weight
    std::vector<Tensor*> parameters() override {
        return {&weight};
    }

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

struct GradPRelu : GradFn {
    Tensor input;
    Tensor weight;
    
    GradPRelu(const Tensor& inp, const Tensor& w) 
        : input(inp), weight(w) {
        parents = {input, weight};
    }
    void backward(const Tensor& self) override;
};