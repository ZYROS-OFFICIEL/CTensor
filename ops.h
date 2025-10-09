#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensors.h"

// helper: produce result shape for elementwise binary op (a and b already padded to same ndim)
static std::vector<size_t> compute_result_shape_padded(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.ndim, b.ndim);
    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        result[i] = std::max(a.shape[i], b.shape[i]);
    }
    return result;
}

// ---------------- elementwise ops (use read_scalar_at/write_scalar_at) ----------------

Tensor add_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.dtype, false);

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
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        double va = read_scalar_at(a.data, index_a, a.dtype);
        double vb = read_scalar_at(b.data, index_b, b.dtype);
        write_scalar_at(result.data, flat, result.dtype, va + vb);
    }

    return result;
}

Tensor diff_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        double va = read_scalar_at(a.data, index_a, a.dtype);
        double vb = read_scalar_at(b.data, index_b, b.dtype);
        write_scalar_at(result.data, flat, result.dtype, va - vb);
    }

    return result;
}

Tensor mul_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        double va = read_scalar_at(a.data, index_a, a.dtype);
        double vb = read_scalar_at(b.data, index_b, b.dtype);
        write_scalar_at(result.data, flat, result.dtype, va * vb);
    }

    return result;
}

Tensor div_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        double va = read_scalar_at(a.data, index_a, a.dtype);
        double vb = read_scalar_at(b.data, index_b, b.dtype);
        write_scalar_at(result.data, flat, result.dtype, va / vb);
    }

    return result;
}

Tensor pow_(const Tensor& a_, const Tensor& b_) {
    size_t ndim_result = std::max(a_.ndim, b_.ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    std::vector<size_t> result_shape = compute_result_shape_padded(a, b);
    Tensor result(result_shape, a.dtype, false);

    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.strides[i];
            index_b += idx_b * b.strides[i];
        }

        double va = read_scalar_at(a.data, index_a, a.dtype);
        double vb = read_scalar_at(b.data, index_b, b.dtype);
        write_scalar_at(result.data, flat, result.dtype, std::pow(va, vb));
    }

    return result;
}

// ---------------- batched / broadcasted matmul ----------------
// A: (..., m, k)  B: (..., k, n) => C: (..., m, n)
Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.ndim < 1 || B.ndim < 1)
        throw std::invalid_argument("matmul: tensors must have at least 1 dimension");

    // shapes & strides pointers available in Tensor
    // extract m, k, n
    size_t kA = A.shape[A.ndim - 1];
    size_t kB = (B.ndim >= 2) ? B.shape[B.ndim - 2] : 1;
    if (kA != kB) throw std::invalid_argument("matmul: inner dimensions do not match.");

    size_t m = (A.ndim >= 2) ? A.shape[A.ndim - 2] : 1;
    size_t n = (B.ndim >= 2) ? B.shape[B.ndim - 1] : 1;

    // batch dims vectors (left to right)
    std::vector<size_t> batchA;
    if (A.ndim > 2) batchA = std::vector<size_t>(A.shape, A.shape + (A.ndim - 2));
    std::vector<size_t> batchB;
    if (B.ndim > 2) batchB = std::vector<size_t>(B.shape, B.shape + (B.ndim - 2));

    std::vector<size_t> batchShape = broadcast_batch_shape_from_vectors(batchA, batchB);

    // output shape
    std::vector<size_t> outShape = batchShape;
    outShape.push_back(m);
    outShape.push_back(n);
    Tensor C(outShape, A.dtype, false);

    // convenience
    size_t batchRank = batchShape.size();
    size_t totalBatch = 1;
    for (auto s : batchShape) totalBatch *= s;

    // precompute multipliers for batch multi-index conversion
    std::vector<size_t> batchMul(batchRank, 1);
    for (int i = (int)batchRank - 2; i >= 0; --i) batchMul[i] = batchMul[i+1] * batchShape[i+1];

    // compute base offset (in elements) for A/B given broadcast-aware batchIndex
    auto compute_base_offset = [&](const Tensor& T, const std::vector<size_t>& T_batch_shape,
                                   const std::vector<size_t>& fullBatchShape,
                                   const std::vector<size_t>& batchIndex) -> size_t {
        size_t offset = 0;
        size_t pad = (fullBatchShape.size() > T_batch_shape.size()) ? (fullBatchShape.size() - T_batch_shape.size()) : 0;
        for (size_t d = 0; d < fullBatchShape.size(); ++d) {
            size_t dimT = (d < pad) ? 1 : T_batch_shape[d - pad];
            size_t idxT = (dimT == 1) ? 0 : batchIndex[d];
            if (d >= pad) offset += idxT * T.strides[d - pad];
        }
        return offset;
    };

    std::vector<size_t> batchIndex(batchRank, 0);
    for (size_t batch = 0; batch < totalBatch; ++batch) {
        // batch -> multi-index
        size_t rem = batch;
        for (size_t d = 0; d < batchRank; ++d) {
            batchIndex[d] = rem / batchMul[d];
            rem = rem % batchMul[d];
        }

        size_t baseA = compute_base_offset(A, batchA, batchShape, batchIndex);
        size_t baseB = compute_base_offset(B, batchB, batchShape, batchIndex);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t t = 0; t < kA; ++t) {
                    size_t offA = baseA + i * A.strides[A.ndim - 2] + t * A.strides[A.ndim - 1];
                    size_t offB = baseB + t * B.strides[B.ndim - 2] + j * B.strides[B.ndim - 1];
                    double va = read_scalar_at(A.data, offA, A.dtype);
                    double vb = read_scalar_at(B.data, offB, B.dtype);
                    sum += va * vb;
                }
                // write to C at (batchIndex..., i, j)
                size_t offC = 0;
                for (size_t d = 0; d < batchRank; ++d) offC += batchIndex[d] * C.strides[d];
                offC += i * C.strides[batchRank];
                offC += j * C.strides[batchRank + 1];
                write_scalar_at(C.data, offC, C.dtype, sum);
            }
        }
    }

    return C;
}

// operator overloads (keep these)
Tensor operator+(const Tensor& a, const Tensor& b) { return add_(a,b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return diff_(a,b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul_(a,b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div_(a,b); }
Tensor operator^(const Tensor& a, const Tensor& b) { return pow_(a,b); }

// NOTE: operator@ is removed because C++ has no '@' operator.
// Use matmul(a,b) instead.
