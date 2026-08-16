#pragma once
#include "core.h"
#include "neuralnet.h"
#include <vector>
#include <stdexcept>

namespace nn {


class Embedding : public Module {
public:
    int num_embeddings;
    int embedding_dim;
    int padding_idx;
    
    Tensor weight;

    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1) 
        : num_embeddings(num_embeddings), 
          embedding_dim(embedding_dim),
          padding_idx(padding_idx)
    {
        if (padding_idx >= num_embeddings || padding_idx < -1) {
            throw std::invalid_argument("padding_idx must be within num_embeddings");
        }

        weight = Tensor::empty({num_embeddings, embedding_dim});
        normal_(weight, 0.0, 1.0);

        if (padding_idx >= 0) {
            Tensor pad_row = weight.select(0, padding_idx);
            zeros_(pad_row);
        }

        weight.requires_grad_(true);
    }

    Tensor forward(const Tensor& indices) {
        return functional::embedding(weight, indices, padding_idx);
    }

    Tensor operator()(const Tensor& indices) {
        return forward(indices);
    }

    std::vector<Tensor*> parameters() override {
        return { &weight };
    }

    NamedParams named_parameters(const std::string& prefix = "") override {
        return { {prefix + "weight", &weight} };
    }
};

} 