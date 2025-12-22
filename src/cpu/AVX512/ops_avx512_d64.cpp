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
// Note: pow requires specialized implementation That handle later as it's complex
Tensor pow_avx512_d64(const Tensor& a, const Tensor& b) {
    // Basic broadcasting logic handled via MP-style loop for this specific op 
    // because binary_op_broadcast takes intrinsics. We implement a specific broadcast loop here.
    
    std::vector<size_t> out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape, a.device(), DType::Double64);
    
    // We reuse the broadcast structure but with scalar fallback inside SIMD loop
    // This allows the compiler to vectorise `pow(a,b)` if possible.
    
    // Simplification: Delegate to generic logic if shapes differ, OR:
    // We can't easily pass a lambda with `std::pow` to binary_op_broadcast_512_d64 
    // because it expects __m512d.
    // So we use a scalar loop with #pragma omp simd.
    
    size_t out_numel = out.numel();
    double* out_ptr = (double*)out.data();
    const double* a_ptr = (const double*)a.data();
    const double* b_ptr = (const double*)b.data();
    
    auto out_mult = build_index_multipliers(out_shape);
    auto a_strides = shape_to_strides_bytes(a.shape());
    auto b_strides = shape_to_strides_bytes(b.shape());
    bool a_contig = a.is_contiguous() && a.shape() == out_shape;
    bool b_contig = b.is_contiguous() && b.shape() == out_shape;
    bool a_scalar = a.numel() == 1;
    bool b_scalar = b.numel() == 1;

    #pragma omp parallel for
    for (size_t i = 0; i < out_numel; ++i) {
        double val_a, val_b;
        
        if (a_contig) val_a = a_ptr[i];
        else if (a_scalar) val_a = a_ptr[0];
        else {
            int32_t off = compute_offset_bytes(i, out_shape, out_mult, a.shape(), a_strides);
            // off is bytes, we need double index? No, pointers are double*.
            // Wait, compute_offset_bytes returns BYTE offset.
            val_a = *(const double*)((const char*)a_ptr + off);
        }

        if (b_contig) val_b = b_ptr[i];
        else if (b_scalar) val_b = b_ptr[0];
        else {
            int32_t off = compute_offset_bytes(i, out_shape, out_mult, b.shape(), b_strides);
            val_b = *(const double*)((const char*)b_ptr + off);
        }

        out_ptr[i] = std::pow(val_a, val_b);
    }
    return out;
}
Tensor matmul_avx512_d64(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) throw std::runtime_error("matmul_avx512_d64: only 2D");
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];
    if (K != B.shape()[0]) throw std::runtime_error("matmul_avx512_d64: shape mismatch");

    Tensor C({M, N}, A.device(), DType::Double64);
    const double* a_ptr = (const double*)A.data();
    const double* b_ptr = (const double*)B.data();
    double* c_ptr = (double*)C.data();
    std::memset(c_ptr, 0, M * N * sizeof(double));

    // Blocked loop for AVX-512 (8 doubles)
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m512d va = _mm512_set1_pd(a_ptr[i*K + k]);
            
            size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m512d vc = _mm512_loadu_pd(c_ptr + i*N + j);
                __m512d vb = _mm512_loadu_pd(b_ptr + k*N + j);
                vc = _mm512_fmadd_pd(va, vb, vc);
                _mm512_storeu_pd(c_ptr + i*N + j, vc);
            }
            // Tail
            if (j < N) {
                __mmask8 mask = tail_mask(N - j);
                __m512d vc = _mm512_maskz_loadu_pd(mask, c_ptr + i*N + j);
                __m512d vb = _mm512_maskz_loadu_pd(mask, b_ptr + k*N + j);
                vc = _mm512_fmadd_pd(va, vb, vc);
                _mm512_mask_storeu_pd(c_ptr + i*N + j, mask, vc);
            }
        }
    }
    return C;
}
//                        Comparison Implementations

template<int CMP_PRED>
Tensor cmp_avx512_d64_impl(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512_d64(a, b, [](__m512d x, __m512d y){
        __mmask8 k = _mm512_cmp_pd_mask(x, y, CMP_PRED);
        return _mm512_mask_blend_pd(k, _zmm_0, _zmm_1);
    });
}
Tensor lt_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_LT_OQ>(a,b); }
Tensor le_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_LE_OQ>(a,b); }
Tensor gt_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_GT_OQ>(a,b); }
Tensor ge_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_GE_OQ>(a,b); }
Tensor eq_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx512_d64(const Tensor& a, const Tensor& b) { return cmp_avx512_d64_impl<_CMP_NEQ_OQ>(a,b); }

//                        Unary Operations (Double)

Tensor abs_avx512_d64(const Tensor& a) { 
    return unary_op_512_d64(a, [](__m512d x){ return _mm512_abs_pd(x); }); 
}

Tensor sqrt_avx512_d64(const Tensor& a) { 
    return unary_op_512_d64(a, [](__m512d x){ return _mm512_sqrt_pd(x); }); 
}

Tensor relu_avx512_d64(const Tensor& a) { 
    return unary_op_512_d64(a, [](__m512d x){ return _mm512_max_pd(x, _zmm_0); }); 
}
#define OMP_SIMD_UNARY_D64(FUNC_NAME, STD_FUNC) \
Tensor FUNC_NAME(const Tensor& a) { \
    Tensor out(a.shape(), a.device(), DType::Double64); \
    const double* pa = (const double*)a.data(); \
    double* pout = (double*)out.data(); \
    size_t n = a.numel(); \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < n; ++i) { \
        pout[i] = STD_FUNC(pa[i]); \
    } \
    return out; \
}
