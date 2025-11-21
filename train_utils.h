#pragma once
#include "module.h"
#include <vector>

// --- Generalized Training/Eval Utilities ---

// If you wrap your layers in the Module class (above), you can use this:
inline void set_train_mode(std::vector<Module*>& layers, bool train_mode) {
    for (auto* layer : layers) {
        if (train_mode) layer->train();
        else layer->eval();
    }
}

// Or, if you use a container like Sequential:
inline void set_model_mode(Module& model, bool train_mode) {
    if (train_mode) model.train();
    else model.eval();
}

// --- Optimizer Interface (Basic SGD) ---
// This fits perfectly here as it interacts with the parameters collected from modules.

class Optimizer {
public:
    std::vector<Tensor*> params;
    double lr;

    Optimizer(const std::vector<Tensor*>& parameters, double learning_rate) 
        : params(parameters), lr(learning_rate) {}

    void step() {
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue; // Skip if no grad
            
            // p = p - lr * grad
            // Simple SGD update
            // Note: p->data -= lr * p->grad
            size_t n = p->numel();
            // We need direct access for speed, or use tensor ops
            // Tensor update = *p - tensor_from_grad(*p) * lr;
            // But we want in-place update.
            
            // Manual loop for now (assumes float32/double consistency)
            // Ideally, add a `sub_in_place` or `add_scaled` op to Tensor.
            // For now, using ops1 functions is safest but creates new tensors.
            // Ideally: p->data[i] -= lr * p->grad[i]
            
            // Since we don't have an easy in-place tensor op exposed yet:
            for (size_t i = 0; i < n; ++i) {
                // Use read_scalar_at / write_scalar_at to be safe with types
                // BUT parameters are usually contiguous weights.
                // Assuming contiguous for optimization here is common, but let's be safe.
                // Wait, params are weights created by us, they are contiguous.
                
                double p_val = read_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype);
                double g_val = read_scalar_at(p->impl->storage->grad.get(), i, p->impl->dtype);
                
                write_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype, p_val - lr * g_val);
            }
        }
    }

    void zero_grad() {
        for (auto* p : params) {
            if (p->impl->storage->grad) {
                // Re-allocate or memset to zero
                // ensure_grad_buffer(*p, true) would do it if exposed
                // Or just memset directly.
                size_t nbytes = p->numel() * p->dtype_bytes();
                std::memset(p->impl->storage->grad.get(), 0, nbytes);
            }
        }
    }
};