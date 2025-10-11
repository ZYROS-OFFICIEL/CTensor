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
