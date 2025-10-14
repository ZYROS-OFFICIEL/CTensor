#pragma once
#include "tensors.h"
#include "ops.h"

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    // Compute total number of elements in new shape
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel_())
        throw std::invalid_argument("reshape: total number of elements must remain constant.");

    // Copy current tensor (shallow copy of data)
    Tensor out = *this;

    // Free previous shape & strides
    if (out.shape) free(out.shape);
    if (out.strides) free(out.strides);

    // Allocate new shape & strides
    out.ndim = new_shape.size();
    out.shape = (size_t*) malloc(out.ndim * sizeof(size_t));
    out.strides = (size_t*) malloc(out.ndim * sizeof(size_t));
    if (!out.shape || !out.strides) throw std::bad_alloc();

    // Copy shape values
    for (size_t i = 0; i < out.ndim; ++i) out.shape[i] = new_shape[i];

    // Recompute strides
    if (out.ndim > 0) {
        out.strides[out.ndim - 1] = 1;
        for (int i = (int)out.ndim - 2; i >= 0; --i)
            out.strides[i] = out.strides[i + 1] * out.shape[i + 1];
    }

    return out;
}
Tensor Tensor::select(size_t dim, size_t index) const {
    if (dim >= ndim) 
        throw std::out_of_range("select: dimension out of range");
    if (index >= shape[dim])
        throw std::out_of_range("select: index out of range");

    // Build new shape (remove the selected dimension)
    Tensor out = *this; // shallow copy of data
    if (out.shape) free(out.shape);
    if (out.strides) free(out.strides);

    out.ndim = ndim - 1;
    out.shape = (size_t*) malloc(out.ndim * sizeof(size_t));
    out.strides = (size_t*) malloc(out.ndim * sizeof(size_t));
    if (!out.shape || !out.strides) throw std::bad_alloc();

    // Copy shape & strides, skipping the selected dim
    for (size_t i = 0, j = 0; i < ndim; ++i) {
        if (i == dim) continue;
        out.shape[j] = shape[i];
        out.strides[j] = strides[i];
        ++j;
    }

    // Offset data pointer to selected slice
    out.data = static_cast<char*>(data) + index * strides[dim];

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
    // Build new shape without dimensions of size 1
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] != 1) new_shape.push_back(shape[i]);
    }

    // If all dims were 1 → keep a single 1D scalar shape {1}
    if (new_shape.empty()) new_shape.push_back(1);

    // Create output tensor (shallow copy of data)
    Tensor out = *this;

    // Free old shape & strides
    if (out.shape) free(out.shape);
    if (out.strides) free(out.strides);

    // Allocate new shape & strides
    out.ndim = new_shape.size();
    out.shape = (size_t*) malloc(out.ndim * sizeof(size_t));
    out.strides = (size_t*) malloc(out.ndim * sizeof(size_t));
    if (!out.shape || !out.strides) throw std::bad_alloc();

    // Copy new shape values
    for (size_t i = 0; i < out.ndim; ++i) out.shape[i] = new_shape[i];

    // Recompute strides
    if (out.ndim > 0) {
        out.strides[out.ndim - 1] = 1;
        for (int i = (int)out.ndim - 2; i >= 0; --i)
            out.strides[i] = out.strides[i + 1] * out.shape[i + 1];
    }

    return out;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (dim > ndim)
        throw std::out_of_range("unsqueeze: dimension out of range.");

    // Build new shape with a 1 inserted at 'dim'
    std::vector<size_t> new_shape(ndim + 1);
    for (size_t i = 0, j = 0; i < ndim + 1; ++i) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            new_shape[i] = shape[j++];
        }
    }

    // Create output tensor (shallow copy)
    Tensor out = *this;

    // Free old shape & strides
    if (out.shape) free(out.shape);
    if (out.strides) free(out.strides);

    // Allocate new shape & strides
    out.ndim = new_shape.size();
    out.shape = (size_t*) malloc(out.ndim * sizeof(size_t));
    out.strides = (size_t*) malloc(out.ndim * sizeof(size_t));
    if (!out.shape || !out.strides) throw std::bad_alloc();

    // Copy new shape
    for (size_t i = 0; i < out.ndim; ++i) out.shape[i] = new_shape[i];

    // Recompute strides
    if (out.ndim > 0) {
        out.strides[out.ndim - 1] = 1;
        for (int i = (int)out.ndim - 2; i >= 0; --i)
            out.strides[i] = out.strides[i + 1] * out.shape[i + 1];
    }

    return out;
}

Tensor Tensor::flatten() const {
    Tensor out = *this; // shallow copy of data

    // Free old shape & strides
    if (out.shape) free(out.shape);
    if (out.strides) free(out.strides);

    // Allocate new shape & strides
    out.ndim = 1;
    out.shape = (size_t*) malloc(sizeof(size_t));
    out.strides = (size_t*) malloc(sizeof(size_t));
    if (!out.shape || !out.strides) throw std::bad_alloc();

    out.shape[0] = numel_();
    out.strides[0] = 1;

    return out;
}

