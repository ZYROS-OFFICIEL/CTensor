#include "dropout.h"
#include "ops1.h" // For mul_, etc.
#include <stdexcept>
#include <random>
#include <ctime>

// Constructor
Dropout::Dropout(double p) : p(p) {
    if (p < 0.0 || p >= 1.0) 
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

Tensor Dropout::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Dropout: null input");

    // If p == 0, it's identity
    if (p == 0.0) return input;

    // Create a random binary mask 
    // Mask is 1 with probability (1-p), 0 with probability p
    Tensor mask = Tensor::zeros(input.shape(), input._dtype(), false);
    size_t n = input.numel();
    
    // Simple random generation (could be optimized)
    // Note: In a real library, use a better RNG seeding strategy.
    static std::mt19937 gen(1234); // Fixed seed for reproducibility during dev
    std::bernoulli_distribution d(1.0 - p); // True with prob (1-p)

    // We need to iterate stride-aware to fill the mask?
    // Actually, mask is new and contiguous, so flat iteration is safe for writing.
    // BUT we must write using strided access if we used a view-aware constructor.
    // Tensor::zeros creates a contiguous tensor, so we can write to storage directly linearly.
    auto* m_data = mask.impl->storage->data.get();
    for (size_t i = 0; i < n; ++i) {
        double val = d(gen) ? 1.0 : 0.0;
        write_scalar_at(m_data, i, mask._dtype(), val);
    }

    // Scale factor for Inverted Dropout
    double scale = 1.0 / (1.0 - p);

    // Apply mask and scale: output = input * mask * scale
    // We can use our ops. mask requires_grad=false.
    Tensor output = input * mask;
    output = output * scale;

    // Attach GradFn
    if (input.requires_grad()) {
        // We need to save the mask for the backward pass
        output.impl->grad_fn = std::make_shared<GradDropout>(input, mask, scale);
    }

    return output;
}