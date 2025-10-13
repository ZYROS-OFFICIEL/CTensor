#pragma once
#include "tensors.h"
#include "ops.h"

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel_())
        throw std::invalid_argument("reshape: total number of elements must remain constant.");

    Tensor out = *this;
    out.shape = new_shape;
    return out;
}
Tensor Tensor::arange(double start, double end, double step, DType dtype) {
    size_t n = static_cast<size_t>((end - start) / step);
    Tensor t({n}, dtype, false);
    for (size_t i = 0; i < n; ++i) {
        double v = start + i * step;
        write_scalar_at(t.data, i, dtype, v);
    }
    return t;
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
Tensor Tensor::unsqueeze(size_t dim) const {
    if (dim > shape.size())
        throw std::out_of_range("unsqueeze: dimension out of range.");

    std::vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    Tensor out = *this;
    out.shape = new_shape;
    return out;
}
Tensor Tensor::flatten() const {
    Tensor out = *this;
    out.shape = {numel()};
    return out;
}
