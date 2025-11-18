#include "linear.h"
#include "ops1.h" // for matmul, add
#include <stdexcept>
#include <cmath>

// --- Linear Layer Implementation ---

Linear::Linear(int in_feat, int out_feat, bool with_bias)
    : in_features(in_feat), out_features(out_feat) 
{
    // Initialize weights using Kaiming/He initialization (or simple random)
    // Standard practice: Uniform(-k, k) where k = sqrt(1/in_features)
    double k = std::sqrt(1.0 / in_features);
    
    // Create weights [out, in]
    // Note: We store as [out, in] because matmul(x, w.T) is standard
    weight = Tensor::rand({(size_t)out_features, (size_t)in_features}, DType::Float32, true);
    
    // Better initialization manually (optional, but good for convergence)
    // For now, just use the rand() which is [0, 1], then scale/shift
    // or just leave as random for simple tests.
    // Let's leave as generic random for now to match your Conv style.

    if (with_bias) {
        bias = Tensor::zeros({(size_t)out_features}, DType::Float32, true);
    } else {
        bias = Tensor(); // Empty/null tensor
    }
}

Tensor Linear::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Linear: null input");
    
    // Input shape expected: [Batch, in_features] or [*, in_features]
    // We simply rely on matrix multiplication: Y = X @ W.T + b
    
    // 1. Transpose weights to [in, out] for matmul
    // Tensor::t_() is in-place, so we need a way to get a transposed view or clone.
    // Your library has t_() which modifies internal strides. 
    // Let's create a view that is transposed.
    // Actually, your matmul_ probably expects (M, K) and (K, N).
    // input: (B, in)
    // weight: (out, in) -> transpose -> (in, out)
    // output: (B, out)
    
    // Create a transposed view of weight (non-destructive)
    // We can use permute({1, 0}) if it's 2D.
    if (weight.impl->ndim != 2) throw std::runtime_error("Linear: weight must be 2D");
    Tensor w_t = weight.permute({1, 0});

    // 2. MatMul
    Tensor output = matmul_(input, w_t);

    // 3. Add Bias (if exists)
    if (bias.impl) {
        // Bias is [out]. Output is [B, out].
        // Standard add_ should handle broadcasting if implemented correctly.
        output = output + bias;
    }

    return output;
}



Tensor Linear::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Linear: null input");
    
    // Input shape expected: [Batch, in_features] or [*, in_features]
    // We simply rely on matrix multiplication: Y = X @ W.T + b
    
    // 1. Transpose weights to [in, out] for matmul
    // Tensor::t_() is in-place, so we need a way to get a transposed view or clone.
    // Your library has t_() which modifies internal strides. 
    // Let's create a view that is transposed.
    // Actually, your matmul_ probably expects (M, K) and (K, N).
    // input: (B, in)
    // weight: (out, in) -> transpose -> (in, out)
    // output: (B, out)
    
    // Create a transposed view of weight (non-destructive)
    // We can use permute({1, 0}) if it's 2D.
    if (weight.impl->ndim != 2) throw std::runtime_error("Linear: weight must be 2D");
    Tensor w_t = weight.permute({1, 0});

    // 2. MatMul
    Tensor output = matmul_(input, w_t);

    // 3. Add Bias (if exists)
    if (bias.impl) {
        // Bias is [out]. Output is [B, out].
        // Standard add_ should handle broadcasting if implemented correctly.
        output = output + bias;
    }

    return output;
}


// --- Flatten Layer Implementation ---

Tensor Flatten::forward(const Tensor& input) const {
    if (!input.impl) throw std::runtime_error("Flatten: null input");
    
    std::vector<size_t> shape = input.shape();
    int ndim = (int)shape.size();
    
    // Resolve negative end_dim
    int end = (end_dim < 0) ? (ndim + end_dim) : end_dim;
    
    if (start_dim < 0 || start_dim >= ndim || end < 0 || end >= ndim || start_dim > end) {
        // Fallback or error. For standard CNN usage [B, C, H, W] -> [B, features]
        // start=1, end=-1 implies flattening dimensions 1, 2, 3.
        // Let's just implement the standard "flatten from start_dim to end"
    }

    std::vector<size_t> new_shape;
    
    // Dimensions before start_dim are kept
    for (int i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape[i]);
    }
    
    // Dimensions from start to end are merged
    size_t flattened_size = 1;
    for (int i = start_dim; i <= end; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);
    
    // Dimensions after end_dim are kept
    for (int i = end + 1; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
    }

    // Return reshaped view
    return input.reshape(new_shape);
}