#include "ops_avx2_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>

#if defined(__AVX2__)

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
//                     Internal AVX2 Math Constants & Helpers
// ========================================================================

const __m256 _ps_1  = _mm256_set1_ps(1.0f);
const __m256 _ps_05 = _mm256_set1_ps(0.5f);
const __m256 _ps_0  = _mm256_setzero_ps();
const __m256 _ps_nan= _mm256_set1_ps(NAN);
const __m256i _pi32_0x7f = _mm256_set1_epi32(0x7f);

// --- Masked Load/Store Helpers ---
// Using manual loop for tails is often safer/simpler than _mm256_maskload_ps 
// near page boundaries if mask generation is complex.

static inline __m256 masked_loadu_ps(const float* ptr, size_t valid_count) {
    alignas(32) float tmp[8] = {0.0f};
    for (size_t i = 0; i < valid_count; ++i) tmp[i] = ptr[i];
    return _mm256_load_ps(tmp);
}

static inline void masked_storeu_ps(float* ptr, __m256 v, size_t valid_count) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    for (size_t i = 0; i < valid_count; ++i) ptr[i] = tmp[i];
}

// --- Abs ---
inline __m256 _mm256_abs_ps(__m256 x) {
    const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask));
}

// --- Exponential (Exp) ---
inline __m256 exp256_ps(__m256 x) {
    __m256 tmp = _mm256_setzero_ps();
    __m256 fx;
    __m256i emm0;
    __m256 one = _ps_1;

    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    fx = _mm256_add_ps(fx, _ps_05);
    fx = _mm256_floor_ps(fx);
    
    tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f));
    __m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x, x);
    
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _ps_05);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _pi32_0x7f);
    emm0 = _mm256_slli_epi32(emm0, 23);
    
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

// --- Natural Logarithm (Ln) ---
inline __m256 log256_ps(__m256 x) {
    __m256 one = _ps_1;
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);

    x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f)); 

    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    
    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff)));
    x = _mm256_or_ps(x, _ps_05);

    emm0 = _mm256_sub_epi32(emm0, _pi32_0x7f);
    __m256 e = _mm256_cvtepi32_ps(emm0);
    e = _mm256_add_ps(e, one);

    __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(7.0376836292E-2f);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.1514610310E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.1676998740E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.2420140846E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.4249322787E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.6668057665E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(2.0000714765E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-2.4999993993E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(3.3333331174E-1f));
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    tmp = _mm256_mul_ps(e, _mm256_set1_ps(-2.12194440e-4f));
    y = _mm256_add_ps(y, tmp);

    tmp = _mm256_mul_ps(z, _ps_05);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, _mm256_set1_ps(0.693359375f));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

// --- Sine (Sin) ---
inline __m256 sin256_ps(__m256 x) {
    __m256 xmm1, sign_bit, y;
    __m256i emm2;
    sign_bit = x;
    x = _mm256_abs_ps(x);

    xmm1 = _mm256_mul_ps(x, _mm256_set1_ps(0.63661977236758134308f)); // 2/pi
    emm2 = _mm256_cvttps_epi32(xmm1);
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(emm2, _mm256_set1_epi32(4)), _mm256_setzero_si256()));
    sign_bit = _mm256_xor_ps(sign_bit, _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), poly_mask));

    __m256 m1 = _mm256_mul_ps(y, _mm256_set1_ps(-1.5703125f));
    __m256 m2 = _mm256_mul_ps(y, _mm256_set1_ps(-4.837512969970703125e-4f));
    __m256 m3 = _mm256_mul_ps(y, _mm256_set1_ps(-7.549789948768648e-8f));
    
    x = _mm256_add_ps(x, m1);
    x = _mm256_add_ps(x, m2);
    x = _mm256_add_ps(x, m3);

    __m256 z = _mm256_mul_ps(x, x);
    y = _mm256_set1_ps(2.443315711809948E-005f);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.388731625493765E-003f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_set1_ps(4.166664568298827E-002f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    __m256 tmp = _mm256_mul_ps(z, _ps_05);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, _ps_1);
    y = _mm256_mul_ps(y, x);

    return _mm256_xor_ps(y, _mm256_and_ps(sign_bit, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))));
}

// --- Cosine (Cos) ---
inline __m256 cos256_ps(__m256 x) {
    x = _mm256_add_ps(x, _mm256_set1_ps(1.57079632679489661923f));
    return sin256_ps(x);
}

// --- Tanh ---
inline __m256 tanh256_ps(__m256 x) {
    __m256 two_x = _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    __m256 exp_2x = exp256_ps(two_x);
    __m256 num = _mm256_sub_ps(exp_2x, _ps_1);
    __m256 den = _mm256_add_ps(exp_2x, _ps_1);
    return _mm256_div_ps(num, den);
}

// --- Sigmoid ---
inline __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    __m256 e = exp256_ps(neg_x);
    __m256 den = _mm256_add_ps(_ps_1, e);
    return _mm256_div_ps(_ps_1, den);
}

// --- Pow ---
inline __m256 pow256_ps(__m256 a, __m256 b) {
    return exp256_ps(_mm256_mul_ps(b, log256_ps(a)));
}

// --- Horizontal Sum ---
inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

} // namespace

// ---------------------------
// Broadcasting helpers
// ---------------------------

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

// ----------------------Binary Broadcast Template (AVX2)----------------------

template <typename Func>
Tensor binary_op_broadcast(const Tensor& A, const Tensor& B, Func op) {
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
    bool a_is_scalar = (a_shape.empty() || (a_shape.size() == 1 && a_shape[0] == 1));
    bool b_is_scalar = (b_shape.empty() || (b_shape.size() == 1 && b_shape[0] == 1));

    size_t vec_end = (out_numel / 8) * 8;

    #pragma omp parallel
    {
        int32_t gather_idx[8];

        #pragma omp for
        for (size_t i = 0; i < vec_end; i += 8) {
            __m256 va;
            if (a_contig) {
                va = _mm256_loadu_ps(a_ptr + i);
            } else if (a_is_scalar) {
                va = _mm256_set1_ps(a_ptr[0]);
            } else {
                for (int lane = 0; lane < 8; ++lane)
                    gather_idx[lane] = compute_offset_bytes(i + lane, out_shape, out_mult, a_shape, a_strides);
                va = _mm256_i32gather_ps(a_ptr, _mm256_loadu_si256((const __m256i*)gather_idx), 1);
            }

            __m256 vb;
            if (b_contig) {
                vb = _mm256_loadu_ps(b_ptr + i);
            } else if (b_is_scalar) {
                vb = _mm256_set1_ps(b_ptr[0]);
            } else {
                for (int lane = 0; lane < 8; ++lane)
                    gather_idx[lane] = compute_offset_bytes(i + lane, out_shape, out_mult, b_shape, b_strides);
                vb = _mm256_i32gather_ps(b_ptr, _mm256_loadu_si256((const __m256i*)gather_idx), 1);
            }

            __m256 vr = op(va, vb);
            _mm256_storeu_ps(out_ptr + i, vr);
        }

        // Tail handling (single thread usually fine for tail, but we do it inside parallel if needed, or after)
    }
    
    // Tail loop (scalar/masked)
    if (vec_end < out_numel) {
        size_t tail = out_numel - vec_end;
        
        // A Load
        __m256 va;
        if (a_contig) {
            va = masked_loadu_ps(a_ptr + vec_end, tail);
        } else if (a_is_scalar) {
            va = _mm256_set1_ps(a_ptr[0]);
        } else {
            int32_t gidx[8] = {0};
            for (size_t j=0; j<tail; ++j) 
                gidx[j] = compute_offset_bytes(vec_end + j, out_shape, out_mult, a_shape, a_strides);
            va = _mm256_i32gather_ps(a_ptr, _mm256_loadu_si256((const __m256i*)gidx), 1);
        }

        // B Load
        __m256 vb;
        if (b_contig) {
            vb = masked_loadu_ps(b_ptr + vec_end, tail);
        } else if (b_is_scalar) {
            vb = _mm256_set1_ps(b_ptr[0]);
        } else {
            int32_t gidx[8] = {0};
            for (size_t j=0; j<tail; ++j) 
                gidx[j] = compute_offset_bytes(vec_end + j, out_shape, out_mult, b_shape, b_strides);
            vb = _mm256_i32gather_ps(b_ptr, _mm256_loadu_si256((const __m256i*)gidx), 1);
        }

        __m256 vr = op(va, vb);
        masked_storeu_ps(out_ptr + vec_end, vr, tail);
    }

    return out;
}

//-----------------------Unary Template (AVX2)----------------------

template <typename Func>
Tensor unary_op_broadcast(const Tensor& A, Func op) {
    Tensor out(A.shape(), DType::Float32);
    const float* a_ptr = get_ptr<float>(A);
    float* out_ptr = get_ptr<float>(out);
    size_t n = A.numel();
    size_t vec_end = (n / 8) * 8;

    #pragma omp parallel for
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a_ptr + i);
        __m256 vr = op(va);
        _mm256_storeu_ps(out_ptr + i, vr);
    }
    if (n > vec_end) {
        size_t tail = n - vec_end;
        __m256 va = masked_loadu_ps(a_ptr + vec_end, tail);
        __m256 vr = op(va);
        masked_storeu_ps(out_ptr + vec_end, vr, tail);
    }
    return out;
}

// ========================================================================
//                        Implementations
// ========================================================================

Tensor add_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_add_ps(x, y); });
}
Tensor sub_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_sub_ps(x, y); });
}
Tensor mul_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_mul_ps(x, y); });
}
Tensor div_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_div_ps(x, y); });
}
Tensor pow_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return pow256_ps(x, y); });
}

// Comparisons
template<int CMP_FLAG>
Tensor cmp_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, []( __m256 x, __m256 y){
        __m256 m = _mm256_cmp_ps(x, y, CMP_FLAG);
        return _mm256_and_ps(m, _ps_1);
    });
}

Tensor lt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LT_OQ>(a,b); }
Tensor le_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LE_OQ>(a,b); }
Tensor gt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GT_OQ>(a,b); }
Tensor ge_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GE_OQ>(a,b); }
Tensor eq_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_NEQ_OQ>(a,b); }

// Unary
Tensor abs_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_abs_ps(x); }); }
Tensor sqrt_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_sqrt_ps(x); }); }
Tensor relu_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_max_ps(x, _ps_0); }); }
Tensor ln_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return log256_ps(x); }); }
Tensor exp_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return exp256_ps(x); }); }
Tensor sin_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return sin256_ps(x); }); }
Tensor cos_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return cos256_ps(x); }); }
Tensor tanh_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return tanh256_ps(x); }); }
Tensor sigmoid_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return sigmoid256_ps(x); }); }
Tensor softplus_avx2_f32(const Tensor& a) { 
    return unary_op_broadcast(a, [](__m256 x){ 
        return log256_ps(_mm256_add_ps(_ps_1, exp256_ps(x))); 
    }); 
}

#define OMP_SIMD_UNARY_AVX2(FUNC_NAME, STD_FUNC) \
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

OMP_SIMD_UNARY_AVX2(asin_avx2_f32, std::asin)
OMP_SIMD_UNARY_AVX2(acos_avx2_f32, std::acos)
OMP_SIMD_UNARY_AVX2(tan_avx2_f32, std::tan)
OMP_SIMD_UNARY_AVX2(atan_avx2_f32, std::atan)
OMP_SIMD_UNARY_AVX2(sinh_avx2_f32, std::sinh)
OMP_SIMD_UNARY_AVX2(cosh_avx2_f32, std::cosh)

// Matmul (Simple blocked AVX2)
Tensor matmul_avx2_f32(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) throw std::runtime_error("matmul_avx2: only 2D");
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];
    if (K != B.shape()[0]) throw std::runtime_error("matmul_avx2: shape mismatch");

    Tensor C({M, N}, DType::Float32);
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* c_ptr = get_ptr<float>(C);
    
    std::memset(c_ptr, 0, M * N * sizeof(float));

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 va = _mm256_set1_ps(a_ptr[i*K + k]);
            size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 vc = _mm256_loadu_ps(c_ptr + i*N + j);
                __m256 vb = _mm256_loadu_ps(b_ptr + k*N + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(c_ptr + i*N + j, vc);
            }
            if (j < N) {
                size_t tail = N - j;
                __m256 vc = masked_loadu_ps(c_ptr + i*N + j, tail);
                __m256 vb = masked_loadu_ps(b_ptr + k*N + j, tail);
                vc = _mm256_fmadd_ps(va, vb, vc);
                masked_storeu_ps(c_ptr + i*N + j, vc, tail);
            }
        }
    }
    return C;
}

// Reductions
Tensor sum_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_avx2: only dim=-1");
    size_t n = t.numel();
    const float* data = get_ptr<float>(t);
    float global_sum = 0.0f;

    #pragma omp parallel
    {
        __m256 vsum = _ps_0;
        #pragma omp for nowait
        for (size_t i=0; i < n; i+=8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else v = masked_loadu_ps(data + i, tail);
            vsum = _mm256_add_ps(vsum, v);
        }
        float local_sum = hsum256_ps(vsum);
        #pragma omp atomic
        global_sum += local_sum;
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_sum;
    return out;
}

Tensor mean_avx2_f32(const Tensor& t, int dim) {
    Tensor s = sum_avx2_f32(t, dim);
    float n = static_cast<float>(t.numel());
    ((float*)get_ptr<float>(s))[0] /= n;
    return s;
}

Tensor max_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_avx2: only dim=-1");
    const float* data = get_ptr<float>(t);
    size_t n = t.numel();
    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        #pragma omp for nowait
        for(size_t i=0; i<n; i+=8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else v = masked_loadu_ps(data + i, tail);
            vmax = _mm256_max_ps(vmax, v);
        }
        // Horizontal max
        __m128 vlow = _mm256_castps256_ps128(vmax);
        __m128 vhigh = _mm256_extractf128_ps(vmax, 1);
        vlow = _mm_max_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        vlow = _mm_max_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, vlow);
        vlow = _mm_max_ss(vlow, shuf);
        float local_max = _mm_cvtss_f32(vlow);
        
        #pragma omp critical
        {
            if(local_max > global_max) global_max = local_max;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_max;
    return out;
}

Tensor min_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_avx2: only dim=-1");
    const float* data = get_ptr<float>(t);
    size_t n = t.numel();
    float global_min = std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        #pragma omp for nowait
        for(size_t i=0; i<n; i+=8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else v = masked_loadu_ps(data + i, tail);
            vmin = _mm256_min_ps(vmin, v);
        }
        // Horizontal min
        __m128 vlow = _mm256_castps256_ps128(vmin);
        __m128 vhigh = _mm256_extractf128_ps(vmin, 1);
        vlow = _mm_min_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        vlow = _mm_min_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, vlow);
        vlow = _mm_min_ss(vlow, shuf);
        float local_min = _mm_cvtss_f32(vlow);
        
        #pragma omp critical
        {
            if(local_min < global_min) global_min = local_min;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_min;
    return out;
}

#endif // __AVX2__