#include "neuralnet.h"
#include "core.h" 
#include <stdexcept>
#include <cmath>
#include <iostream>

// --- Linear Layer Implementation ---

Linear::Linear(int in_feat, int out_feat, bool with_bias, DType dt)
    : in_features(in_feat), out_features(out_feat) ,dtype(dt)
{
    // Weight: [out_features, in_features]
    weight = Tensor::rand({(size_t)out_features, (size_t)in_features}, dtype, true);
    
    if (with_bias) {
        bias = Tensor::zeros({(size_t)out_features}, dtype, true);
    } else {
        bias = Tensor(); 
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Linear: null input");

    Tensor w_t = weight.permute({1, 0});
    std::vector<size_t> in_shape = input.shape();

    if (in_shape.size() <= 2) {
        Tensor output = matmul(input, w_t);
        if (bias.impl) output = output + bias;
        return output;
    }

    // matmul only supports exactly-2D operands on real hardware (AVX2/AVX512 kernels
    // throw otherwise) — flatten leading dims to 2D, matmul, then restore them.
    size_t batch = 1;
    for (size_t i = 0; i + 1 < in_shape.size(); ++i) batch *= in_shape[i];
    Tensor flat = input.contiguous().reshape({batch, in_shape.back()});

    Tensor out2d = matmul(flat, w_t);
    if (bias.impl) out2d = out2d + bias;

    std::vector<size_t> out_shape = in_shape;
    out_shape.back() = (size_t)out_features;
    return out2d.reshape(out_shape);
}

// --- Flatten Layer Implementation ---

Tensor Flatten::forward(const Tensor& input) const {
    if (!input.impl) throw std::runtime_error("Flatten: null input");
    
    std::vector<size_t> shape = input.shape();
    int ndim = (int)shape.size();
    
    // Default behavior: Flatten [N, C, H, W] -> [N, C*H*W]
    // start_dim = 1, end_dim = -1
    
    int start = start_dim;
    int end = (end_dim < 0) ? (ndim + end_dim) : end_dim;
    
    if (start < 0 || start >= ndim || end < 0 || end >= ndim || start > end) {
        // Fallback: just return input or throw error
        return input;
    }

    std::vector<size_t> new_shape;
    
    // 1. Dimensions before start_dim are kept (Batch dim)
    for (int i = 0; i < start; ++i) {
        new_shape.push_back(shape[i]);
    }
    
    // 2. Flatten range [start, end]
    size_t flattened_size = 1;
    for (int i = start; i <= end; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);
    
    // 3. Dimensions after end_dim are kept
    for (int i = end + 1; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
    }

    return input.reshape(new_shape);
}