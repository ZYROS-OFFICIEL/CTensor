#include "ops_avx512_d64.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>

#if defined(__AVX512F__)

namespace {
//                     Internal AVX-512 Constants & Helpers

const __m512d _zmm_1  = _mm512_set1_pd(1.0);
const __m512d _zmm_0  = _mm512_setzero_pd();
const __m512d _zmm_nan= _mm512_set1_pd(NAN);

// Helper to generate a mask for the first 'n' elements (0 <= n <= 8)
inline __mmask8 tail_mask(size_t n) {
    return (__mmask8)((1U << n) - 1);
}

// --- Horizontal Sum for ZMM (Double) ---
inline double hsum512_pd(__m512d v) {
    // reduce 512 -> 256
    __m256d vlow = _mm512_castpd512_pd256(v);
    __m256d vhigh = _mm512_extractf64x4_pd(v, 1);
    vlow = _mm256_add_pd(vlow, vhigh);
    // reduce 256 -> 128
    __m128d xlow = _mm256_castpd256_pd128(vlow);
    __m128d xhigh = _mm256_extractf128_pd(vlow, 1);
    xlow = _mm_add_pd(xlow, xhigh);
    // reduce 128 -> 64
    __m128d shuf = _mm_movedup_pd(xlow); // broadcast high to low
    __m128d sums = _mm_add_pd(xlow, shuf);
    return _mm_cvtsd_f64(sums);
}

}
//                     Broadcasting & Dispatch Logic

static inline std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = sizeof(double);
    for (int i = (int)shape.size()-2; i >= 0; --i) {
        strides[i] = strides[i+1] * (int64_t)shape[i+1];
    }
    return strides;
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t ai = (i < n - na) ? 1 : a[i - (n - na)];
        size_t bi = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (ai != 1 && bi != 1 && ai != bi) throw std::runtime_error("broadcast: incompatible shapes");
        out[i] = std::max(ai, bi);
    }
    return out;
}

static std::vector<int64_t> build_index_multipliers(const std::vector<size_t>& shape) {
    std::vector<int64_t> mult(shape.size());
    if (shape.empty()) return mult;
    mult.back() = 1;
    for (int i = (int)shape.size()-2; i >= 0; --i) mult[i] = mult[i+1] * (int64_t)shape[i+1];
    return mult;
}

static inline int32_t compute_offset_bytes(size_t lin_idx, const std::vector<size_t>& out_shape, const std::vector<int64_t>& out_mult, const std::vector<size_t>& in_shape, const std::vector<int64_t>& in_strides_bytes) {
    int32_t offset = 0;
    size_t nd = out_shape.size();
    size_t offset_dim = nd - in_shape.size();
    for (size_t d = 0; d < nd; ++d) {
        size_t coord = (lin_idx / out_mult[d]) % out_shape[d];
        size_t in_coord = 0;
        if (d >= offset_dim) {
            size_t idx = d - offset_dim;
            if (in_shape[idx] != 1) in_coord = coord;
            offset += (int32_t)(in_coord * in_strides_bytes[idx]);
        }
    }
    return offset;
}

//                     Binary Broadcast Template (Double)

Tensor binary_op_broadcast_512_d64(const Tensor& A, const Tensor& B, std::function<__m512d(__m512d,__m512d)> op) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out(out_shape, A.device(), DType::Double64);
    const double* a_ptr = (const double*)A.data();
    const double* b_ptr = (const double*)B.data();
    double* out_ptr = (double*)out.data();

    auto out_mult = build_index_multipliers(out_shape);
    auto a_strides = shape_to_strides_bytes(a_shape);
    auto b_strides = shape_to_strides_bytes(b_shape);

    bool a_contig = A.is_contiguous() && a_shape == out_shape;
    bool b_contig = B.is_contiguous() && b_shape == out_shape;
    bool a_scalar = A.numel() == 1;
    bool b_scalar = B.numel() == 1;

    // Process blocks of 8 doubles (512 bits)
    #pragma omp parallel for
    for (size_t i = 0; i < out_numel; i += 8) {
        size_t rem = out_numel - i;
        __mmask8 k = (rem >= 8) ? 0xFF : tail_mask(rem);

        // Load A
        __m512d va;
        if (a_contig) {
            va = _mm512_maskz_loadu_pd(k, a_ptr + i);
        } else if (a_scalar) {
            va = _mm512_set1_pd(a_ptr[0]);
        } else {
            // Scatter/Gather indices calculation (8 offsets)
            int32_t idx_buf[8];
            for (int l = 0; l < 8; ++l) {
                if ((k >> l) & 1) { 
                    idx_buf[l] = compute_offset_bytes(i + l, out_shape, out_mult, a_shape, a_strides);
                }
            }
            // For doubles, we use a 256-bit index vector (8 x 32-bit ints)
            __m256i vidx = _mm256_loadu_si256((const __m256i*)idx_buf);
            // Scale=1 because offsets are in bytes
            va = _mm512_mask_i32gather_pd(_zmm_0, k, vidx, a_ptr, 1);
        }

        // Load B
        __m512d vb;
        if (b_contig) {
            vb = _mm512_maskz_loadu_pd(k, b_ptr + i);
        } else if (b_scalar) {
            vb = _mm512_set1_pd(b_ptr[0]);
        } else {
            int32_t idx_buf[8];
            for (int l = 0; l < 8; ++l) {
                if ((k >> l) & 1) idx_buf[l] = compute_offset_bytes(i + l, out_shape, out_mult, b_shape, b_strides);
            }
            __m256i vidx = _mm256_loadu_si256((const __m256i*)idx_buf);
            vb = _mm512_mask_i32gather_pd(_zmm_0, k, vidx, b_ptr, 1);
        }

        // Op
        __m512d vr = op(va, vb);

        // Store
        _mm512_mask_storeu_pd(out_ptr + i, k, vr);
    }
    return out;
}

//                  Unary Template (Double)

Tensor unary_op_512_d64(const Tensor& A, std::function<__m512d(__m512d)> op) {
    Tensor out(A.shape(), A.device(), DType::Double64);
    const double* a_ptr = (const double*)A.data();
    double* out_ptr = (double*)out.data();
    size_t n = A.numel();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 8) {
        size_t rem = n - i;
        __mmask8 k = (rem >= 8) ? 0xFF : tail_mask(rem);
        __m512d va = _mm512_maskz_loadu_pd(k, a_ptr + i);
        __m512d vr = op(va);
        _mm512_mask_storeu_pd(out_ptr + i, k, vr);
    }
    return out;
}

//                        Binary Implementations

Tensor add_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512_d64(a, b, [](__m512d x, __m512d y){ return _mm512_add_pd(x, y); });
}
Tensor sub_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512_d64(a, b, [](__m512d x, __m512d y){ return _mm512_sub_pd(x, y); });
}
Tensor mul_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512_d64(a, b, [](__m512d x, __m512d y){ return _mm512_mul_pd(x, y); });
}
Tensor div_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512_d64(a, b, [](__m512d x, __m512d y){ return _mm512_div_pd(x, y); });
}
