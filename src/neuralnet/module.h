#pragma once
#include "core/tensor.h"
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

// A container that holds a list of modules
class Sequential : public Module {
public:
    std::vector<std::shared_ptr<Module>> modules;

    void add(std::shared_ptr<Module> module) {
        modules.push_back(module);
    }

    // Recursive train/eval
    void train() override {
        training = true;
        for (auto& m : modules) m->train();
    }

    void eval() override {
        training = false;
        for (auto& m : modules) m->eval();
    }
    
    // Recursive parameter collection
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> params;
        for (auto& m : modules) {
            auto p = m->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
    
};