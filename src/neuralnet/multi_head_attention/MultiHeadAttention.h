#pragma once
#include "core.h"
#include "neuralnet.h"
#include <cmath>
#include <stdexcept>

namespace nn {

class MultiheadAttention : public Module {
public:
    int embed_dim;
    int num_heads;
    int head_dim;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    MultiheadAttention(int embed_dim, int num_heads, bool bias = true) 
        : embed_dim(embed_dim), 
          num_heads(num_heads), 
          head_dim(embed_dim / num_heads),
          q_proj(embed_dim, embed_dim, bias),
          k_proj(embed_dim, embed_dim, bias),
          v_proj(embed_dim, embed_dim, bias),
          out_proj(embed_dim, embed_dim, bias) 
    {
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("embed_dim must be perfectly divisible by num_heads");
        }
    }
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_mask = Tensor()) {
        int bsz = query.shape()[0];
        int q_len = query.shape()[1];
        int k_len = key.shape()[1];
        int v_len = value.shape()[1];

        // 1. Apply initial Linear projections
        Tensor q = q_proj(query);
        Tensor k = k_proj(key);
        Tensor v = v_proj(value);

        q = q.reshape({bsz, q_len, num_heads, head_dim}).permute({0, 2, 1, 3});
        k = k.reshape({bsz, k_len, num_heads, head_dim}).permute({0, 2, 1, 3});
        v = v.reshape({bsz, v_len, num_heads, head_dim}).permute({0, 2, 1, 3});

        Tensor attn_output = scaled_dot_product_attention(q, k, v, attn_mask);

        attn_output = attn_output.permute({0, 2, 1, 3});
        
        attn_output = attn_output.contiguous().reshape({bsz, q_len, embed_dim});

        return out_proj(attn_output);
    }

    Tensor operator()(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_mask = Tensor()) {
        return forward(query, key, value, attn_mask);
    }

    std::vector<Tensor*> parameters() override {
        return combine_params(q_proj, k_proj, v_proj, out_proj);
    }
};

} 