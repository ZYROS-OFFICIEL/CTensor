#pragma once
#include <vector>
#include <functional>
#include "tensor.h"

struct Transforme {
    // pipeline of tensor transformations
    std::vector<std::function<Tensor(const Tensor&)>> pipeline;

    // ----- Add a normalization operation to the pipeline -----
    void normalize_(const std::vector<float>& mean, const std::vector<float>& stdv);

    // ----- Add a resize operation to the pipeline -----
    void resize_(size_t H, size_t W);

    void to_(DType dtype);

    // ----- Apply all transformations in the pipeline -----
    Tensor operator()(const Tensor& input) const;
};
