#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensor1.h"

// helper: produce result shape for elementwise binary op (a and b already padded to same ndim)
static std::vector<size_t> compute_result_shape_padded(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.impl->ndim, b.impl->ndim);
    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        result[i] = std::max(a.impl->shape[i], b.impl->shape[i]);
    }
    return result;
}

// ---------------- elementwise ops (use read_scalar_at/write_scalar_at) ----------------

Tensor add_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.impl->dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using strides (with broadcasting)
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va + vb);
    }

    return result;
}

Tensor diff_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.impl->dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using strides (with broadcasting)
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va - vb);
    }

    return result;
}