#pragma once
#include "core.h"
#include "neuralnet.h"
#include <cmath>
#include <stdexcept>

inline Tensor scaled_dot_product_attention(
    const Tensor& query, 
    const Tensor& key, 
    const Tensor& value, 
    const Tensor& attn_mask = Tensor(), 
    double dropout_p = 0.0
) {
    int head_dim = query.shape().back();

    std::vector<size_t> perm_dims;
    for (size_t i = 0; i < key.shape().size(); ++i) {
        perm_dims.push_back(i);
    }
    std::swap(perm_dims[perm_dims.size() - 2], perm_dims[perm_dims.size() - 1]);
    Tensor k_t = key.permute(perm_dims);

    Tensor attn_scores = matmul(query, k_t);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_scores = mul_scalar(attn_scores, scale);

    if (attn_mask.impl) { 
        attn_scores = add(attn_scores, attn_mask);
    }

    Tensor attn_weights = softmax(attn_scores, -1);

    return matmul(attn_weights, value);
}
