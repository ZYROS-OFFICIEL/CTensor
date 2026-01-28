#pragma once
#include "core/tensor.h"
#include "core/autograd.h"
#include "neuralnet/module.h" 

// --- Dropout Layer ---
// Randomly zeros some of the elements of the input tensor with probability p.
// Used for regularization during training.
class Dropout : public Module {
public:
    double p; // probability of an element to be zeroed
    
    Dropout(double p = 0.5);

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};

// --- Autograd Node ---
struct GradDropout : GradFn {
    Tensor input;
    Tensor mask; // Binary mask used in forward pass
    double scale;

    GradDropout(const Tensor& input_, const Tensor& mask_, double scale_) 
        : input(input_), mask(mask_), scale(scale_) {
        parents = {input};
    }
    
    void backward(const Tensor& self) override;
};