#include "neuralnet.h"

LayerNorm::LayerNorm(int normalized_shape, double eps_) : eps(eps_) {
    weight = Tensor::ones({(size_t)normalized_shape}, DType::Float32, true);
    bias = Tensor::zeros({(size_t)normalized_shape}, DType::Float32, true);
}

Tensor LayerNorm::forward(const Tensor& x) {
    size_t last_dim = x.shape().size() - 1;

    Tensor mu = mean(x, -1).unsqueeze(last_dim);
    Tensor centered = x - mu;
    Tensor var = mean(centered * centered, -1).unsqueeze(last_dim);
    Tensor normed = centered / sqrt(var + eps);
    return normed * weight + bias;
}
