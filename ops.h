#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensors.h"

std::vector<size_t> add_shape(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.ndim, b.ndim);
    std::vector<size_t> shape(ndim, 1);

    for (int i = 0; i < ndim; i++) {
        size_t dim_a = (i >= ndim - a.ndim) ? a.shape[i - (ndim - a.ndim)] : 1;
        size_t dim_b = (i >= ndim - b.ndim) ? b.shape[i - (ndim - b.ndim)] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            throw std::runtime_error("Shapes not compatible for broadcasting");

        shape[i] = std::max(dim_a, dim_b);
    }
    return shape;
}

Tensor add_(const Tensor& a_, const Tensor& b_) {
    // 1️⃣ Pad tensors to same ndim
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // 2️⃣ Determine result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; i++) {
        result_shape[i] = std::max(a.shape[i], b.shape[i]);
    }

    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(ndim_result, 0);

    // 3️⃣ Loop over all elements
    for (size_t flat = 0; flat < n; flat++) {
        // compute multi-dimensional index
        size_t rem = flat;
        for (int i = ndim_result - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; i++) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);

            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        result.data[flat] = a.data[index_a] + b.data[index_b];
    }

    return result;
}


Tensor mul_(const Tensor& a_, const Tensor& b_) {
    // 1️⃣ Pad tensors to same ndim
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // 2️⃣ Determine result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; i++) {
        result_shape[i] = std::max(a.shape[i], b.shape[i]);
    }

    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(ndim_result, 0);

    // 3️⃣ Loop over all elements
    for (size_t flat = 0; flat < n; flat++) {
        // compute multi-dimensional index
        size_t rem = flat;
        for (int i = ndim_result - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; i++) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);

            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        result.data[flat] = a.data[index_a] * b.data[index_b];
    }

    return result;
}

Tensor div_(const Tensor& a_, const Tensor& b_) {
    // 1️⃣ Pad tensors to same ndim
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // 2️⃣ Determine result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; i++) {
        result_shape[i] = std::max(a.shape[i], b.shape[i]);
    }

    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(ndim_result, 0);

    // 3️⃣ Loop over all elements
    for (size_t flat = 0; flat < n; flat++) {
        // compute multi-dimensional index
        size_t rem = flat;
        for (int i = ndim_result - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; i++) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);

            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        result.data[flat] = a.data[index_a] / b.data[index_b];
    }

    return result;
}
Tensor pow_(const Tensor& a_, const Tensor& b_) {
    // 1️⃣ Pad tensors to same ndim
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // 2️⃣ Determine result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; i++) {
        result_shape[i] = std::max(a.shape[i], b.shape[i]);
    }

    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(ndim_result, 0);

    // 3️⃣ Loop over all elements
    for (size_t flat = 0; flat < n; flat++) {
        // compute multi-dimensional index
        size_t rem = flat;
        for (int i = ndim_result - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; i++) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);

            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        result.data[flat] = std::pow(a.data[index_a], b.data[index_b]);
    }

    return result;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return add_(a,b); 
}
Tensor operator*(const Tensor& a, const Tensor& b) {
    return mul_(a,b); 
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return div_(a,b); 
}

Tensor operator^(const Tensor& a, const Tensor& b) {
    return pow_(a,b); 
}