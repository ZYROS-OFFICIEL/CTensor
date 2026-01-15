#pragma once
#include "tensor.h"
#include <vector>
#include <string>
#include <memory>

class Module {
public:
    bool training;

    Module() : training(true) {}
    virtual ~Module() = default;

    // Basic mode switching
    virtual void train() { training = true; }
    virtual void eval() { training = false; }

    // Forward pass interface (optional, as some layers have specific signatures)
    // virtual Tensor forward(const Tensor& input) = 0;
    
    // Helper to get all trainable parameters (weights/biases)
    // This is crucial for the Optimizer!
    virtual std::vector<Tensor*> parameters() {
        return {}; // Default implementation returns nothing
    }
};
