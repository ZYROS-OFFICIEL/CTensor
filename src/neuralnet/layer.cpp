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
    Tensor output = matmul(input, w_t);

    // 3. Add Bias (if exists)
    if (bias.impl) {
        // Bias is [out]. Output is [Batch, out].
        output = output + bias;
    }

    return output;
}
