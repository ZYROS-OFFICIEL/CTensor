#include "dropout.h"
#include "core/ops_dispatch.h"
#include <stdexcept>
#include <random>
#include <ctime>

Dropout::Dropout(double p) : p(p) {
    if (p < 0.0 || p > 1.0) 
        throw std::invalid_argument("Dropout: p must be in [0, 1]");
}

Tensor Dropout::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Dropout: null input");

    if (!training || p == 0.0) {
        return input; // Identity in eval mode
    }

    Tensor mask = Tensor::zeros(input.shape(), input._dtype(), false);
    size_t n = input.numel();
    
    double scale = (p == 1.0) ? 0.0 : (1.0 / (1.0 - p));

    auto* m_data = mask.impl->data->data.get();
    
    if (p < 1.0) {
        static std::mt19937 gen(1234); 
        std::bernoulli_distribution d(1.0 - p); 

        for (size_t i = 0; i < n; ++i) {
            double val = d(gen) ? 1.0 : 0.0;
            write_scalar_at(m_data, i, mask._dtype(), val);
        }
    }
    Tensor output = input * mask;
    output = output * scale;

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradDropout>(input, mask, scale);
    }

    return output;
}

void GradDropout::backward(const Tensor& self) {
    if (!self.impl->grad->data)
        throw std::runtime_error("GradDropout: missing self grad");
    
    if (input.requires_grad()) {
        Tensor grad_output = tensor_from_grad(self);
        
        Tensor grad_input = grad_output * mask;
        grad_input = grad_input * scale;

        accumulate_grad(input, grad_input);
    }
}