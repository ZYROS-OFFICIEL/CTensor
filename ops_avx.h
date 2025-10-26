#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensor1.h"
#include <immintrin.h>

// helper: produce result shape for elementwise binary op (a and b already padded to same ndim)
static std::vector<size_t> compute_result_shape_padded(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.impl->ndim, b.impl->ndim);
    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        result[i] = std::max(a.impl->shape[i], b.impl->shape[i]);
    }
    return result;
}

//------------------Helper ---------------------------------------------
static bool broadcastable(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);

    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        if (da != db && da != 1 && db != 1)
            return false;
    }
    return true;
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);
    std::vector<size_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("Incompatible shapes for broadcasting");
        result[i] = std::max(da, db);
    }
    return result;
}
// ---------------- elementwise ops (use read_scalar_at/write_scalar_at) ----------------
Tensor add_simd(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("null tensor");

    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // compute result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; ++i) {
        size_t da = (i < ndim_result - a.impl->ndim) ? 1 : a.impl->shape[i - (ndim_result - a.impl->ndim)];
        size_t db = (i < ndim_result - b.impl->ndim) ? 1 : b.impl->shape[i - (ndim_result - b.impl->ndim)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("shapes not broadcastable");
        result_shape[i] = std::max(da, db);
    }

    Tensor result(result_shape, a.impl->dtype, false);

    size_t n = result.numel_();

    // Check if both are contiguous along the last dimension for SIMD
    bool a_contig = (a.impl->ndim == 1 || a.impl->strides[a.impl->ndim-1] == 1);
    bool b_contig = (b.impl->ndim == 1 || b.impl->strides[b.impl->ndim-1] == 1);

    double* out = reinterpret_cast<double*>(result.impl->storage->data.get());
    double* p_a = reinterpret_cast<double*>(a.impl->storage->data.get());
    double* p_b = reinterpret_cast<double*>(b.impl->storage->data.get());

    size_t i = 0;
    const size_t simd_width = 4; // AVX2: 4 doubles per 256-bit register

    // vectorized loop
    for (; i + simd_width <= n; i += simd_width) {
        __m256d va, vb, vc;

        // load values, handle broadcasting manually
        va = _mm256_loadu_pd(p_a + (a_contig ? i : 0));
        vb = _mm256_loadu_pd(p_b + (b_contig ? i : 0));
        vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(out + i, vc);
    }

    // leftover scalar
    for (; i < n; ++i) {
        double va_s = a_contig ? p_a[i] : read_scalar_at(p_a, i, a.impl->dtype);
        double vb_s = b_contig ? p_b[i] : read_scalar_at(p_b, i, b.impl->dtype);
        write_scalar_at(out, i, result.impl->dtype, va_s + vb_s);
    }

    return result;
}
Tensor diff_avx2(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("null tensor");

    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // compute result shape
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; ++i) {
        size_t da = (i < ndim_result - a.impl->ndim) ? 1 : a.impl->shape[i - (ndim_result - a.impl->ndim)];
        size_t db = (i < ndim_result - b.impl->ndim) ? 1 : b.impl->shape[i - (ndim_result - b.impl->ndim)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("shapes not broadcastable");
        result_shape[i] = std::max(da, db);
    }

    Tensor result(result_shape, a.impl->dtype, false);

    size_t n = result.numel_();

    // Check if both are contiguous along the last dimension for SIMD
    bool a_contig = (a.impl->ndim == 1 || a.impl->strides[a.impl->ndim-1] == 1);
    bool b_contig = (b.impl->ndim == 1 || b.impl->strides[b.impl->ndim-1] == 1);

    double* out = reinterpret_cast<double*>(result.impl->storage->data.get());
    double* p_a = reinterpret_cast<double*>(a.impl->storage->data.get());
    double* p_b = reinterpret_cast<double*>(b.impl->storage->data.get());

    size_t i = 0;
    const size_t simd_width = 4; // AVX2: 4 doubles per 256-bit register

    // vectorized loop
    for (; i + simd_width <= n; i += simd_width) {
        __m256d va, vb, vc;

        // load values, handle broadcasting manually
        va = _mm256_loadu_pd(p_a + (a_contig ? i : 0));
        vb = _mm256_loadu_pd(p_b + (b_contig ? i : 0));
        vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(out + i, vc);
    }

    // leftover scalar
    for (; i < n; ++i) {
        double va_s = a_contig ? p_a[i] : read_scalar_at(p_a, i, a.impl->dtype);
        double vb_s = b_contig ? p_b[i] : read_scalar_at(p_b, i, b.impl->dtype);
        write_scalar_at(out, i, result.impl->dtype, va_s + vb_s);
    }

    return result;
}