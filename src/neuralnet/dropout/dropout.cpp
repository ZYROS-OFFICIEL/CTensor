#include "dropout.h"
#include "ops.h"
#include <stdexcept>
#include <random>
#include <ctime>

Dropout::Dropout(double p) : p(p) {
    if (p < 0.0 || p >= 1.0) 
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

Tensor Dropout::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Dropout: null input");

    // Check inherited training flag
    if (!training || p == 0.0) {
        return input; // Identity in eval mode
    }

    // --- TRAINING MODE ---

    // Create a random binary mask 
    // Mask is 1 with probability (1-p), 0 with probability p
    Tensor mask = Tensor::zeros(input.shape(), input._dtype(), false);
    size_t n = input.numel();
    
    // Simple random generation (could be optimized)
    static std::mt19937 gen(1234); // Fixed seed for reproducibility during dev
    std::bernoulli_distribution d(1.0 - p); // True with prob (1-p)

    auto* m_data = mask.impl->data->data.get();
    for (size_t i = 0; i < n; ++i) {
        double val = d(gen) ? 1.0 : 0.0;
        write_scalar_at(m_data, i, mask._dtype(), val);
    }

    // Scale factor for Inverted Dropout
    double scale = 1.0 / (1.0 - p);

    // Apply mask and scale: output = input * mask * scale
    Tensor output = input * mask;
    output = output * scale;

    // Attach GradFn
    if (input.requires_grad()) {
        // We need to save the mask for the backward pass
        output.impl->grad_fn = std::make_shared<GradDropout>(input, mask, scale);
    }

    return output;
}

void GradDropout::backward(const Tensor& self) {
    if (!self.impl->grad->data)
        throw std::runtime_error("GradDropout: missing self grad");
    
    if (input.requires_grad()) {
        // Backward is: grad_input = grad_output * mask * scale
        Tensor grad_output = tensor_from_grad(self);
        
        // We use the saved mask and scale
        Tensor grad_input = grad_output * mask;
        grad_input = grad_input * scale;

        accumulate_grad(input, grad_input);
    }
}