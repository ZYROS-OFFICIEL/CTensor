#pragma once
#include "tensors.h"
#include "ops.h"

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel())
        throw std::invalid_argument("reshape: total number of elements must remain constant.");

    Tensor out = *this;
    out.shape = new_shape;
    return out;
}
Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;
    for (auto s : shape)
        if (s != 1) new_shape.push_back(s);

    // If all dims were 1 → keep a single 1D scalar shape {}
    if (new_shape.empty()) new_shape.push_back(1);

    Tensor out = *this;
    out.shape = new_shape;
    return out;
}
