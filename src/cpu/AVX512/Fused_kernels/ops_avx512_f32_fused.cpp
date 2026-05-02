#include "cpu/AVX512/Fused_kernels/ops_avx512_f32_fused.h"
#include "cpu/AVX512ops_avx512_f32.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>
 
#if defined(__AVX512F__)
 
namespace {
 
template<typename T>
inline T* get_ptr(const Tensor& t) {
    return (T*)t.impl->data->data.get() + t.impl->offset;
}
 
#define ZMM_0_PS  _mm512_setzero_ps()
#define ZMM_1_PS  _mm512_set1_ps(1.0f)
#define ZMM_05_PS _mm512_set1_ps(0.5f)
 
inline __mmask16 tail_mask(size_t n) { return (__mmask16)((1U << n) - 1); }
 
inline __m512 bitwise_xor(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
inline __m512 bitwise_and(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
inline __m512 bitwise_or(__m512 a, __m512 b) {
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
 
inline __m512 abs_ps(__m512 x) {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF)));
}
 
inline __m512 exp512_ps(__m512 x) {
    __m512 fx, one = ZMM_1_PS;
    x  = _mm512_min_ps(x, _mm512_set1_ps( 88.3762626647949f));
    x  = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), ZMM_05_PS);
    fx = _mm512_floor_ps(fx);
    __m512 tmp = _mm512_mul_ps(fx, _mm512_set1_ps(0.693359375f));
    __m512 z   = _mm512_mul_ps(fx, _mm512_set1_ps(-2.12194440e-4f));
    x  = _mm512_sub_ps(x, tmp);
    x  = _mm512_sub_ps(x, z);
    z  = _mm512_mul_ps(x, x);
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
 
inline __m512 log512_ps(__m512 x) {
    __m512 one = ZMM_1_PS;
    __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, ZMM_0_PS, _CMP_LE_OQ);
    x = _mm512_max_ps(x, _mm512_set1_ps(1.17549435e-38f));
    __m512i emm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
    x = bitwise_and(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffff)));
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
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 1.1676998740E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.2420140846E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 1.4249322787E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.6668057665E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 2.0000714765E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-2.4999993993E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 3.3333331174E-1f));
    y = _mm512_mul_ps(y, x);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fmadd_ps(e, _mm512_set1_ps(-2.12194440e-4f), y);
    y = _mm512_fnmadd_ps(z, ZMM_05_PS, y);
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, _mm512_set1_ps(0.693359375f), x);
    return _mm512_mask_blend_ps(invalid_mask, x, _mm512_set1_ps(NAN));
}
 
inline __m512 sigmoid512_ps(__m512 x) {
    __m512 neg_x = bitwise_xor(x, _mm512_set1_ps(-0.0f));
    return _mm512_div_ps(ZMM_1_PS, _mm512_add_ps(ZMM_1_PS, exp512_ps(neg_x)));
}
 
inline __m512 tanh512_ps(__m512 x) {
    __m512 two_x  = _mm512_mul_ps(x, _mm512_set1_ps(2.0f));
    __m512 exp_2x = exp512_ps(two_x);
    return _mm512_div_ps(_mm512_sub_ps(exp_2x, ZMM_1_PS),
                         _mm512_add_ps(exp_2x, ZMM_1_PS));
}
 
inline __m512 relu512_ps(__m512 x) { return _mm512_max_ps(x, ZMM_0_PS); }
 
inline __m512 silu512_ps(__m512 x) { return _mm512_mul_ps(x, sigmoid512_ps(x)); }
 
inline __m512 gelu512_ps(__m512 x) {
    return _mm512_mul_ps(x, sigmoid512_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.702f))));
}
 
inline float hsum512_ps(__m512 v) {
    __m256 lo  = _mm512_castps512_ps256(v);
    __m256 hi  = _mm512_extractf32x8_ps(v, 1);
    __m256 sum = _mm256_add_ps(lo, hi);
    __m128 a   = _mm256_castps256_ps128(sum);
    __m128 b   = _mm256_extractf128_ps(sum, 1);
    a = _mm_add_ps(a, b);
    __m128 shuf = _mm_movehdup_ps(a);
    a = _mm_add_ps(a, shuf);
    shuf = _mm_movehl_ps(shuf, a);
    return _mm_cvtss_f32(_mm_add_ss(a, shuf));
}
 
static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size(), n = std::max(na, nb);
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t ai = (i < n - na) ? 1 : a[i - (n - na)];
        size_t bi = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (ai != 1 && bi != 1 && ai != bi)
            throw std::runtime_error("broadcast: incompatible shapes");
        out[i] = std::max(ai, bi);
    }
    return out;
}
 
static std::vector<int64_t> build_index_multipliers(const std::vector<size_t>& shape) {
    std::vector<int64_t> m(shape.size());
    if (shape.empty()) return m;
    m.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        m[i] = m[i + 1] * (int64_t)shape[i + 1];
    return m;
}
 
static std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> s(shape.size());
    if (shape.empty()) return s;
    s.back() = sizeof(float);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        s[i] = s[i + 1] * (int64_t)shape[i + 1];
    return s;
}
 
static inline int32_t compute_offset_bytes(size_t lin_idx,
                                           const std::vector<size_t>& out_shape,
                                           const std::vector<int64_t>& out_mult,
                                           const std::vector<size_t>& in_shape,
                                           const std::vector<int64_t>& in_strides) {
    int32_t offset = 0;
    size_t nd = out_shape.size(), od = nd - in_shape.size();
    for (size_t d = 0; d < nd; ++d) {
        size_t coord = (lin_idx / (size_t)out_mult[d]) % out_shape[d];
        if (d >= od && in_shape[d - od] != 1)
            offset += (int32_t)(coord * (size_t)in_strides[d - od]);
    }
    return offset;
}

template<typename Func>
Tensor binary_fused_512(const Tensor& A, const Tensor& B, Func op) {
    auto as = A.shape(), bs = B.shape();
    auto os = broadcast_shape(as, bs);
    size_t n = 1;
    for (auto s : os) n *= s;
    Tensor out(os, DType::Float32);
    const float* ap  = get_ptr<float>(A);
    const float* bp  = get_ptr<float>(B);
    float*       op_ = get_ptr<float>(out);
    auto om  = build_index_multipliers(os);
    auto ast = shape_to_strides_bytes(as);
    auto bst = shape_to_strides_bytes(bs);
    bool ac = A.is_contiguous() && as == os;
    bool bc = B.is_contiguous() && bs == os;
    bool as_ = A.numel() == 1;
    bool bs_ = B.numel() == 1;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16) {
        size_t rem = n - i;
        __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
        __m512 va, vb;
        if      (ac)  va = _mm512_maskz_loadu_ps(k, ap + i);
        else if (as_) va = _mm512_set1_ps(ap[0]);
        else {
            int32_t buf[16] = {};
            for (int l = 0; l < 16 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, as, ast);
            va = _mm512_mask_i32gather_ps(ZMM_0_PS, k, _mm512_loadu_si512(buf), ap, 1);
        }
        if      (bc)  vb = _mm512_maskz_loadu_ps(k, bp + i);
        else if (bs_) vb = _mm512_set1_ps(bp[0]);
        else {
            int32_t buf[16] = {};
            for (int l = 0; l < 16 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, bs, bst);
            vb = _mm512_mask_i32gather_ps(ZMM_0_PS, k, _mm512_loadu_si512(buf), bp, 1);
        }
        _mm512_mask_storeu_ps(op_ + i, k, op(va, vb));
    }
    return out;
}
 
template<typename Func>
Tensor ternary_fused_512(const Tensor& A, const Tensor& B, const Tensor& C, Func op) {
    auto as = A.shape(), bs = B.shape(), cs = C.shape();
    auto os = broadcast_shape(broadcast_shape(as, bs), cs);
    size_t n = 1;
    for (auto s : os) n *= s;
    Tensor out(os, DType::Float32);
    const float* ap  = get_ptr<float>(A);
    const float* bp  = get_ptr<float>(B);
    const float* cp  = get_ptr<float>(C);
    float*       op_ = get_ptr<float>(out);
    auto om  = build_index_multipliers(os);
    auto ast = shape_to_strides_bytes(as);
    auto bst = shape_to_strides_bytes(bs);
    auto cst = shape_to_strides_bytes(cs);
    bool ac = A.is_contiguous() && as == os, as_ = A.numel() == 1;
    bool bc = B.is_contiguous() && bs == os, bs_ = B.numel() == 1;
    bool cc = C.is_contiguous() && cs == os, cs_ = C.numel() == 1;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16) {
        size_t rem = n - i;
        __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
        auto load = [&](const float* ptr, bool contig, bool scalar,
                        const std::vector<size_t>& shape,
                        const std::vector<int64_t>& strides) -> __m512 {
            if (contig)  return _mm512_maskz_loadu_ps(k, ptr + i);
            if (scalar)  return _mm512_set1_ps(ptr[0]);
            int32_t buf[16] = {};
            for (int l = 0; l < 16 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, shape, strides);
            return _mm512_mask_i32gather_ps(ZMM_0_PS, k, _mm512_loadu_si512(buf), ptr, 1);
        };
        __m512 va = load(ap, ac, as_, as, ast);
        __m512 vb = load(bp, bc, bs_, bs, bst);
        __m512 vc = load(cp, cc, cs_, cs, cst);
        _mm512_mask_storeu_ps(op_ + i, k, op(va, vb, vc));
    }
    return out;
}
 
template<typename Func>
Tensor unary_fused_512(const Tensor& A, Func op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Float32);
    const float* ap  = get_ptr<float>(A);
    float*       op_ = get_ptr<float>(out);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16) {
        size_t rem = n - i;
        __mmask16 k = (rem >= 16) ? 0xFFFF : tail_mask(rem);
        __m512 va = _mm512_maskz_loadu_ps(k, ap + i);
        _mm512_mask_storeu_ps(op_ + i, k, op(va));
    }
    return out;
}

Tensor fma_avx512_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512 x, __m512 y, __m512 z) {
        return _mm512_fmadd_ps(x, y, z);
    });
}
 
Tensor fms_avx512_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512 x, __m512 y, __m512 z) {
        return _mm512_fmsub_ps(x, y, z);
    });
}
 
Tensor nfma_avx512_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512 x, __m512 y, __m512 z) {
        return _mm512_fnmadd_ps(x, y, z);
    });
}
 
Tensor add_scale_avx512_f32(const Tensor& a, const Tensor& b, float scale) {
    __m512 vs = _mm512_set1_ps(scale);
    return binary_fused_512(a, b, [vs](__m512 x, __m512 y) {
        return _mm512_mul_ps(_mm512_add_ps(x, y), vs);
    });
}
 
Tensor add_relu_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return relu512_ps(_mm512_add_ps(x, y));
    });
}
 
Tensor add_sigmoid_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return sigmoid512_ps(_mm512_add_ps(x, y));
    });
}
 
Tensor add_tanh_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return tanh512_ps(_mm512_add_ps(x, y));
    });
}
 
Tensor mul_add_avx512_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_avx512_f32(a, b, c);
}
 
 
Tensor add_exp_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return exp512_ps(_mm512_add_ps(x, y));
    });
}
 
Tensor add_ln_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return log512_ps(_mm512_add_ps(x, y));
    });
}
 
Tensor exp_neg_avx512_f32(const Tensor& a) {
    return unary_fused_512(a, [](__m512 x) {
        return exp512_ps(bitwise_xor(x, _mm512_set1_ps(-0.0f)));
    });
}
 
Tensor ln_relu_avx512_f32(const Tensor& a) {
    return unary_fused_512(a, [](__m512 x) {
        return log512_ps(_mm512_max_ps(x, ZMM_0_PS));
    });
}
 
Tensor sigmoid_ln_avx512_f32(const Tensor& a) {
    return unary_fused_512(a, [](__m512 x) {
        return log512_ps(sigmoid512_ps(x));
    });
}
 
Tensor silu_avx512_f32(const Tensor& a) {
    return unary_fused_512(a, [](__m512 x) {
        return silu512_ps(x);
    });
}
 
Tensor gelu_avx512_f32(const Tensor& a) {
    return unary_fused_512(a, [](__m512 x) {
        return gelu512_ps(x);
    });
}
 
Tensor swiglu_avx512_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512 x, __m512 y) {
        return _mm512_mul_ps(silu512_ps(x), y);
    });
}
 

}
 
#endif