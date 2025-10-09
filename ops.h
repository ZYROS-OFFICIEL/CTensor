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
Tensor diff_(const Tensor& a_, const Tensor& b_) {
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

        result.data[flat] = a.data[index_a] - b.data[index_b];
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

Tensor matmul(const Tensor& A, const Tensor& B) {
    auto shapeA = A.shape();  // e.g. (2, 3, 4, 5)
    auto shapeB = B.shape();  // e.g. (3, 4, 5, 6)

    if (shapeA.size() < 1 || shapeB.size() < 1)
        throw std::invalid_argument("matmul: tensors must have at least 1 dimension.");

    // Extract matrix dimensions
    size_t m, kA, kB, n;
    kA = shapeA.back();
    n = shapeB.back();
    kB = (shapeB.size() >= 2) ? shapeB[shapeB.size() - 2] : 1;

    if (shapeA.back() != kB)
        throw std::invalid_argument("matmul: inner dimensions do not match.");

    // Batch shapes
    std::vector<size_t> batchA(shapeA.begin(), shapeA.end() - 2);
    std::vector<size_t> batchB(shapeB.begin(), shapeB.end() - 2);
    auto batchShape = broadcast_batch_shape(batchA, batchB);

    m = (shapeA.size() >= 2) ? shapeA[shapeA.size() - 2] : 1;
    n = (shapeB.size() >= 2) ? shapeB.back() : 1;

    // Final output shape
    std::vector<size_t> outShape = batchShape;
    outShape.push_back(m);
    outShape.push_back(n);

    Tensor C(outShape, A.dtype());

    // Compute total batch size
    size_t totalBatch = 1;
    for (auto s : batchShape) totalBatch *= s;

    // Flattened view for iteration
    for (size_t batch = 0; batch < totalBatch; ++batch) {
        // Resolve broadcast indices
        size_t idxA = batch % A.batch_size();
        size_t idxB = batch % B.batch_size();

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t t = 0; t < kA; ++t)
                    sum += A(batch, i, t) * B(batch, t, j);
                C(batch, i, j) = sum;
            }
        }
    }

    return C;
}


Tensor operator+(const Tensor& a, const Tensor& b) {
    return add_(a,b); 
}
Tensor operator-(const Tensor& a, const Tensor& b) {
    return diff_(a,b); 
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
Tensor operator@(const Tensor& a, const Tensor& b) {
    return matmul(a,b); 
}