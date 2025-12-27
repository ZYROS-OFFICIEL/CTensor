#include "ops_avx512_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>
#include <functional> 

#if defined(__AVX512F__)

namespace {

// ========================================================================
//                     Helpers for Tensor Access
// ========================================================================

template <typename T>
inline T* get_ptr(const Tensor& t) {
    if (!t.impl || !t.impl->data) return nullptr;
    return (T*)t.impl->data->data.get() + t.impl->offset;
}

// ========================================================================
//                     AVX-512 Math Helpers (AVX512F Compatible)
// ========================================================================

// FIX: Macros without semicolons to prevent SIGILL at startup and syntax errors
#define ZMM_1_PS   _mm512_set1_ps(1.0f)
#define ZMM_05_PS  _mm512_set1_ps(0.5f)
#define ZMM_0_PS   _mm512_setzero_ps()
#define ZMM_NAN_PS _mm512_set1_ps(NAN)

// --- Bitwise Ops for AVX512F ---
// AVX512F does not have _mm512_and_ps (requires AVX512DQ). 
// We must cast to integer, perform op, and cast back.

inline __m512 bitwise_and(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

inline __m512 bitwise_or(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

inline __m512 bitwise_xor(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

// Helper to generate a mask for the first 'n' elements
inline __mmask16 tail_mask(size_t n) {
    return (__mmask16)((1U << n) - 1);
}

// --- Abs ---
inline __m512 abs_ps(__m512 x) {
    // 0x7FFFFFFF clears the sign bit
    __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), mask));
}

// --- Exponential (Exp) ---
inline __m512 exp512_ps(__m512 x) {
    __m512 fx, one = ZMM_1_PS;
    
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));

    fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), ZMM_05_PS);
    fx = _mm512_floor_ps(fx);
    
    __m512 tmp = _mm512_mul_ps(fx, _mm512_set1_ps(0.693359375f));
    __m512 z   = _mm512_mul_ps(fx, _mm512_set1_ps(-2.12194440e-4f));
    x = _mm512_sub_ps(x, tmp);
    x = _mm512_sub_ps(x, z);
    z = _mm512_mul_ps(x, x);
    
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, ZMM_05_PS);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    __m512i emm0 = _mm512_cvttps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    emm0 = _mm512_slli_epi32(emm0, 23);
    
    return _mm512_mul_ps(y, _mm512_castsi512_ps(emm0));
}

// --- Logarithm (Ln) ---
inline __m512 log512_ps(__m512 x) {
    __m512 one = ZMM_1_PS;
    __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, ZMM_0_PS, _CMP_LE_OQ);
    
    x = _mm512_max_ps(x, _mm512_set1_ps(1.17549435e-38f));

    __m512i emm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
    
    // FIX: Replaced _mm512_and_ps with bitwise_and
    x = bitwise_and(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffff)));
    // FIX: Replaced _mm512_or_ps with bitwise_or
    x = bitwise_or(x, ZMM_05_PS);

    emm0 = _mm512_sub_epi32(emm0, _mm512_set1_epi32(0x7f));
    __m512 e = _mm512_cvtepi32_ps(emm0);
    e = _mm512_add_ps(e, one);

    __mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    
    __m512 tmp = _mm512_maskz_mov_ps(mask, x); 
    
    x = _mm512_mask_sub_ps(x, mask, x, one);
    e = _mm512_mask_sub_ps(e, mask, e, one);
    x = _mm512_mask_add_ps(x, mask, x, tmp);

    __m512 z = _mm512_mul_ps(x, x);
    __m512 y = _mm512_set1_ps(7.0376836292E-2f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.1514610310E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.1676998740E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.2420140846E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.4249322787E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.6668057665E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(2.0000714765E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-2.4999993993E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(3.3333331174E-1f));
    y = _mm512_mul_ps(y, x);
    y = _mm512_mul_ps(y, z);

    y = _mm512_fmadd_ps(e, _mm512_set1_ps(-2.12194440e-4f), y);
    y = _mm512_fnmadd_ps(z, ZMM_05_PS, y); 
    
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, _mm512_set1_ps(0.693359375f), x);

    return _mm512_mask_blend_ps(invalid_mask, x, ZMM_NAN_PS); 
}

// --- Sine (Sin) ---
inline __m512 sin512_ps(__m512 x) {
    __m512 xmm1, sign_bit, y;
    __m512i emm2;
    sign_bit = x;
    x = abs_ps(x);

    xmm1 = _mm512_mul_ps(x, _mm512_set1_ps(0.63661977236758134308f)); 
    emm2 = _mm512_cvttps_epi32(xmm1);
    emm2 = _mm512_add_epi32(emm2, _mm512_set1_epi32(1));
    emm2 = _mm512_and_si512(emm2, _mm512_set1_epi32(~1));
    y = _mm512_cvtepi32_ps(emm2);

    __mmask16 poly_mask = _mm512_cmpeq_epi32_mask(_mm512_and_si512(emm2, _mm512_set1_epi32(4)), _mm512_setzero_si512());
    
    __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
    __m512 xor_mask = _mm512_mask_blend_ps(poly_mask, sign_mask, ZMM_0_PS); 
    
    // FIX: Replaced _mm512_xor_ps with bitwise_xor
    sign_bit = bitwise_xor(sign_bit, xor_mask);

    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-1.5703125f), x);
    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-4.837512969970703125e-4f), x);
    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-7.549789948768648e-8f), x);

    __m512 z = _mm512_mul_ps(x, x);
    y = _mm512_set1_ps(2.443315711809948E-005f);
    y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(-1.388731625493765E-003f));
    y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(4.166664568298827E-002f));
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, ZMM_05_PS, y); 
    y = _mm512_add_ps(y, ZMM_1_PS);
    y = _mm512_mul_ps(y, x);

    // FIX: Replaced _mm512_xor_ps with bitwise_xor
    // Also replaced _mm512_and_ps with bitwise_and
    return bitwise_xor(y, bitwise_and(sign_bit, sign_mask));
}

inline __m512 cos512_ps(__m512 x) {
    x = _mm512_add_ps(x, _mm512_set1_ps(1.57079632679489661923f));
    return sin512_ps(x);
}

inline __m512 tanh512_ps(__m512 x) {
    __m512 two_x = _mm512_mul_ps(x, _mm512_set1_ps(2.0f));
    __m512 exp_2x = exp512_ps(two_x);
    __m512 num = _mm512_sub_ps(exp_2x, ZMM_1_PS);
    __m512 den = _mm512_add_ps(exp_2x, ZMM_1_PS);
    return _mm512_div_ps(num, den);
}

inline __m512 sigmoid512_ps(__m512 x) {
    // FIX: Replaced _mm512_xor_ps with bitwise_xor
    __m512 neg_x = bitwise_xor(x, _mm512_set1_ps(-0.0f));
    __m512 e = exp512_ps(neg_x);
    __m512 den = _mm512_add_ps(ZMM_1_PS, e);
    return _mm512_div_ps(ZMM_1_PS, den);
}

inline __m512 pow512_ps(__m512 a, __m512 b) {
    return exp512_ps(_mm512_mul_ps(b, log512_ps(a)));
}

// Horizontal Sum
inline float hsum512_ps(__m512 v) {
    __m256 vlow = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    vlow = _mm256_add_ps(vlow, vhigh);
    __m128 xlow = _mm256_castps256_ps128(vlow);
    __m128 xhigh = _mm256_extractf128_ps(vlow, 1);
    xlow = _mm_add_ps(xlow, xhigh);
    __m128 shuf = _mm_movehdup_ps(xlow);
    __m128 sums = _mm_add_ps(xlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

} // namespace

// ---------------------------Broadcasting & Dispatch Logic---------------------------

static inline std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = sizeof(float);
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

// ----------------------Binary Broadcast Template----------------------

template <typename Func>
Tensor binary_op_broadcast_512(const Tensor& A, const Tensor& B, Func op) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out(out_shape, DType::Float32);
    
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* out_ptr = get_ptr<float>(out);

    auto out_mult = build_index_multipliers(out_shape);
    auto a_strides = shape_to_strides_bytes(a_shape);
    auto b_strides = shape_to_strides_bytes(b_shape);

    bool a_contig = A.is_contiguous() && a_shape == out_shape;
    bool b_contig = B.is_contiguous() && b_shape == out_shape;
    bool a_scalar = A.numel() == 1;
    bool b_scalar = B.numel() == 1;

    #pragma omp parallel for
    for (size_t i = 0; i < out_numel; i += 16) {
        size_t rem = out_numel - i;
        __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);

        // Load A
        __m512 va;
        if (a_contig) {
            va = _mm512_maskz_loadu_ps(k, a_ptr + i);
        } else if (a_scalar) {
            va = _mm512_set1_ps(a_ptr[0]);
        } else {
            int32_t idx_buf[16];
            for (int l = 0; l < 16; ++l) {
                if ((k >> l) & 1) { 
                    idx_buf[l] = compute_offset_bytes(i + l, out_shape, out_mult, a_shape, a_strides);
                }
            }
            __m512i vidx = _mm512_loadu_si512(idx_buf);
            va = _mm512_mask_i32gather_ps(ZMM_0_PS, k, vidx, a_ptr, 1);
        }

        // Load B
        __m512 vb;
        if (b_contig) {
            vb = _mm512_maskz_loadu_ps(k, b_ptr + i);
        } else if (b_scalar) {
            vb = _mm512_set1_ps(b_ptr[0]);
        } else {
            int32_t idx_buf[16];
            for (int l = 0; l < 16; ++l) {
                if ((k >> l) & 1) idx_buf[l] = compute_offset_bytes(i + l, out_shape, out_mult, b_shape, b_strides);
            }
            __m512i vidx = _mm512_loadu_si512(idx_buf);
            vb = _mm512_mask_i32gather_ps(ZMM_0_PS, k, vidx, b_ptr, 1);
        }

        // Op
        __m512 vr = op(va, vb);

        // Store
        _mm512_mask_storeu_ps(out_ptr + i, k, vr);
    }
    return out;
}

//-----------------------Unary Template----------------------

template <typename Func>
Tensor unary_op_512(const Tensor& A, Func op) {
    Tensor out(A.shape(), DType::Float32);
    
    const float* a_ptr = get_ptr<float>(A);
    float* out_ptr = get_ptr<float>(out);
    size_t n = A.numel();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 16) {
        size_t rem = n - i;
        __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
        __m512 va = _mm512_maskz_loadu_ps(k, a_ptr + i);
        __m512 vr = op(va);
        _mm512_mask_storeu_ps(out_ptr + i, k, vr);
    }
    return out;
}


// ========================================================================
//                        Implementations
// ========================================================================

Tensor add_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, [](__m512 x, __m512 y){ return _mm512_add_ps(x, y); });
}
Tensor sub_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, [](__m512 x, __m512 y){ return _mm512_sub_ps(x, y); });
}
Tensor mul_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, [](__m512 x, __m512 y){ return _mm512_mul_ps(x, y); });
}
Tensor div_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, [](__m512 x, __m512 y){ return _mm512_div_ps(x, y); });
}
Tensor pow_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, [](__m512 x, __m512 y){ return pow512_ps(x, y); });
}


//--------------------------MATMUL--------------------------
Tensor matmul_avx512_f32(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) throw std::runtime_error("matmul_avx512: only 2D");
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];
    if (K != B.shape()[0]) throw std::runtime_error("matmul_avx512: shape mismatch");

    Tensor C({M, N}, DType::Float32);
    
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* c_ptr = get_ptr<float>(C);
    
    std::memset(c_ptr, 0, M * N * sizeof(float));

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m512 va = _mm512_set1_ps(a_ptr[i*K + k]);
            
            size_t j = 0;
            for (; j + 16 <= N; j += 16) {
                __m512 vc = _mm512_loadu_ps(c_ptr + i*N + j);
                __m512 vb = _mm512_loadu_ps(b_ptr + k*N + j);
                vc = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(c_ptr + i*N + j, vc);
            }
            if (j < N) {
                __mmask16 mask = tail_mask(N - j);
                __m512 vc = _mm512_maskz_loadu_ps(mask, c_ptr + i*N + j);
                __m512 vb = _mm512_maskz_loadu_ps(mask, b_ptr + k*N + j);
                vc = _mm512_fmadd_ps(va, vb, vc);
                _mm512_mask_storeu_ps(c_ptr + i*N + j, mask, vc);
            }
        }
    }
    return C;
}

// Comparisons
template<int CMP_PRED>
Tensor cmp_avx512_f32_impl(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_512(a, b, []( __m512 x, __m512 y){
        __mmask16 k = _mm512_cmp_ps_mask(x, y, CMP_PRED);
        return _mm512_mask_blend_ps(k, ZMM_0_PS, ZMM_1_PS);
    });
}

Tensor lt_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_LT_OQ>(a,b); }
Tensor le_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_LE_OQ>(a,b); }
Tensor gt_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_GT_OQ>(a,b); }
Tensor ge_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_GE_OQ>(a,b); }
Tensor eq_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx512_f32(const Tensor& a, const Tensor& b) { return cmp_avx512_f32_impl<_CMP_NEQ_OQ>(a,b); }

// Unary
Tensor abs_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return abs_ps(x); }); }
Tensor sqrt_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return _mm512_sqrt_ps(x); }); }
Tensor relu_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return _mm512_max_ps(x, ZMM_0_PS); }); }
Tensor ln_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return log512_ps(x); }); }
Tensor exp_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return exp512_ps(x); }); }
Tensor sin_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return sin512_ps(x); }); }
Tensor cos_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return cos512_ps(x); }); }
Tensor tanh_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return tanh512_ps(x); }); }
Tensor sigmoid_avx512_f32(const Tensor& a) { return unary_op_512(a, [](__m512 x){ return sigmoid512_ps(x); }); }
Tensor softplus_avx512_f32(const Tensor& a) { 
    return unary_op_512(a, [](__m512 x){ 
        return log512_ps(_mm512_add_ps(ZMM_1_PS, exp512_ps(x))); 
    }); 
}

#define OMP_SIMD_UNARY_512(FUNC_NAME, STD_FUNC) \
Tensor FUNC_NAME(const Tensor& a) { \
    Tensor out(a.shape(), DType::Float32); \
    const float* pa = get_ptr<float>(a); \
    float* pout = get_ptr<float>(out); \
    size_t n = a.numel(); \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < n; ++i) { \
        pout[i] = STD_FUNC(pa[i]); \
    } \
    return out; \
}

OMP_SIMD_UNARY_512(asin_avx512_f32, std::asin)
OMP_SIMD_UNARY_512(acos_avx512_f32, std::acos)
OMP_SIMD_UNARY_512(tan_avx512_f32, std::tan)
OMP_SIMD_UNARY_512(atan_avx512_f32, std::atan)
OMP_SIMD_UNARY_512(sinh_avx512_f32, std::sinh)
OMP_SIMD_UNARY_512(cosh_avx512_f32, std::cosh)

// Reductions
Tensor sum_avx512_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_avx512: only dim=-1");
    size_t n = t.numel();
    const float* data = get_ptr<float>(t);
    float global_sum = 0.0f;

    #pragma omp parallel
    {
        __m512 vsum = ZMM_0_PS;
        #pragma omp for nowait
        for (size_t i=0; i < n; i+=16) {
            size_t rem = n - i;
            __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
            __m512 v = _mm512_maskz_loadu_ps(k, data + i);
            vsum = _mm512_add_ps(vsum, v);
        }
        float local_sum = hsum512_ps(vsum);
        #pragma omp atomic
        global_sum += local_sum;
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_sum;
    return out;
}

Tensor mean_avx512_f32(const Tensor& t, int dim) {
    Tensor s = sum_avx512_f32(t, dim);
    float n = static_cast<float>(t.numel());
    ((float*)get_ptr<float>(s))[0] /= n;
    return s;
}

Tensor max_avx512_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_avx512: only dim=-1");
    const float* data = get_ptr<float>(t);
    size_t n = t.numel();
    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        #pragma omp for nowait
        for(size_t i=0; i<n; i+=16) {
            size_t rem = n - i;
            __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
            __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(-std::numeric_limits<float>::infinity()), k, data+i);
            vmax = _mm512_max_ps(vmax, v);
        }
        float local_max = _mm512_reduce_max_ps(vmax);
        
        #pragma omp critical
        {
            if(local_max > global_max) global_max = local_max;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_max;
    return out;
}

Tensor min_avx512_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_avx512: only dim=-1");
    const float* data = get_ptr<float>(t);
    size_t n = t.numel();
    float global_min = std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        __m512 vmin = _mm512_set1_ps(std::numeric_limits<float>::infinity());
        #pragma omp for nowait
        for(size_t i=0; i<n; i+=16) {
            size_t rem = n - i;
            __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
            __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(std::numeric_limits<float>::infinity()), k, data+i);
            vmin = _mm512_min_ps(vmin, v);
        }
        float local_min = _mm512_reduce_min_ps(vmin);
        
        #pragma omp critical
        {
            if(local_min < global_min) global_min = local_min;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_min;
    return out;
}

#endif // __AVX512F__