#pragma once
#include "tensor.h"
#include "autograd.h"
#include "module.h" 

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
