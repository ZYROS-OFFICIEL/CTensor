#pragma once
#include <vector>
#include <functional>
#include <stdexcept>
#include <iostream>
#include "tensor1.h"  // make sure this has your Tensor class

struct Transforme {
    // Internally store a list of functions that take a Tensor and return a Tensor
    std::vector<std::function<Tensor(const Tensor&)>> pipeline;

    // Add normalization transform (like torchvision.transforms.Normalize)
    void normalize_(const std::vector<float>& mean, const std::vector<float>& stdv) {
        pipeline.push_back([mean, stdv](const Tensor& input) -> Tensor {
            if (input.ndim() < 2)
                throw std::invalid_argument("normalize_: input must have at least 2 dims (C,H,...)");

            size_t C = input.shape()[0];
            if (mean.size() != C && mean.size() != 1)
                throw std::invalid_argument("normalize_: mean length must match channels or be 1");
            if (stdv.size() != C && stdv.size() != 1)
                throw std::invalid_argument("normalize_: std length must match channels or be 1");

            Tensor output = input.clone();
            size_t inner_size = 1;
            for (size_t i = 1; i < input.ndim(); ++i)
                inner_size *= input.shape()[i];

            for (size_t c = 0; c < C; ++c) {
                float m = mean.size() == 1 ? mean[0] : mean[c];
                float s = stdv.size() == 1 ? stdv[0] : stdv[c];
                for (size_t j = 0; j < inner_size; ++j) {
                    size_t idx = c * inner_size + j;
                    double v = input[idx];
                    output[idx] = (v - m) / s;
                }
            }
            return output;
        });
    }

    // Resize placeholder (for now just changes shape, no interpolation)
    void resize_(size_t H, size_t W) {
        pipeline.push_back([H, W](const Tensor& input) -> Tensor {
            Tensor output({input.shape()[0], H, W}, input._dtype());
            return output;
        });
    }

    // Apply all transformations sequentially
    Tensor operator()(const Tensor& input) const {
        Tensor x = input;
        for (const auto& f : pipeline)
            x = f(x);
        return x;
    }
};
