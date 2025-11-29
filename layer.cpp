#include "layer.h"
#include "opsmp.h" // Use optimized ops
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
    
    // Input shape expected: [Batch, in_features]
    
    // 1. Transpose weights to [in, out] for matmul
    Tensor w_t = weight.permute({1, 0});

    // 2. MatMul
    // A=[Batch, in], B=[in, out] -> Result=[Batch, out]
    Tensor output = matmul_mp(input, w_t);

    // 3. Add Bias (if exists)
    if (bias.impl) {
        // Bias is [out]. Output is [Batch, out].
        output = output + bias;
    }

    return output;
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