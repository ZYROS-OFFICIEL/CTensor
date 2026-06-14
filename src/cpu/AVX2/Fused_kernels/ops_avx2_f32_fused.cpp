#include "cpu/AVX2/Fused_kernels/ops_avx2_f32_fused.h"
#include "cpu/AVX2/ops_avx2_f32.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>
 
#if defined(__AVX2__)
 
namespace {
 
template<typename T>
inline T* get_ptr(const Tensor& t) {
    return (T*)t.impl->data->data.get() + t.impl->offset;
}

inline __m256i tail_mask_ps(size_t rem) {
    static const int32_t mask_table[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    return _mm256_loadu_si256((const __m256i*)&mask_table[8 - rem]);
}
 
inline __m256 abs_ps(__m256 x) {
    return _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}
 
inline __m256 exp256_ps(__m256 x) {
    __m256 fx, one = _mm256_set1_ps(1.0f);
    x  = _mm256_min_ps(x, _mm256_set1_ps( 88.3762626647949f));
    x  = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    __m256 tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f));
    __m256 z   = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
    x  = _mm256_sub_ps(x, tmp);
    x  = _mm256_sub_ps(x, z);
    z  = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}
 
inline __m256 log256_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f));
    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff)));
    x = _mm256_or_ps(x, _mm256_set1_ps(0.5f));
    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    __m256 e = _mm256_cvtepi32_ps(emm0);
    e = _mm256_add_ps(e, one);
    __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    __m256 tmp = _mm256_and_ps(mask, x);
    x = _mm256_blendv_ps(x, _mm256_sub_ps(x, one), mask);
    e = _mm256_blendv_ps(e, _mm256_sub_ps(e, one), mask);
    x = _mm256_blendv_ps(x, _mm256_add_ps(x, tmp), mask);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(7.0376836292E-2f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.1514610310E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps( 1.1676998740E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.2420140846E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps( 1.4249322787E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.6668057665E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps( 2.0000714765E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.4999993993E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps( 3.3333331174E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.12194440e-4f), y);
    y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5f), y);
    x = _mm256_add_ps(x, y);
    x = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693359375f), x);
    return _mm256_blendv_ps(x, _mm256_set1_ps(NAN), invalid_mask);
}
 
inline __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), exp256_ps(neg_x)));
}
 
inline __m256 tanh256_ps(__m256 x) {
    __m256 two_x  = _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    __m256 exp_2x = exp256_ps(two_x);
    return _mm256_div_ps(_mm256_sub_ps(exp_2x, _mm256_set1_ps(1.0f)),
                         _mm256_add_ps(exp_2x, _mm256_set1_ps(1.0f)));
}
 
inline __m256 relu256_ps(__m256 x) { return _mm256_max_ps(x, _mm256_setzero_ps()); }
 
inline __m256 silu256_ps(__m256 x) { return _mm256_mul_ps(x, sigmoid256_ps(x)); }
 
inline __m256 gelu256_ps(__m256 x) {
    return _mm256_mul_ps(x, sigmoid256_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.702f))));
}
 
inline float hsum256_ps(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, shuf));
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
Tensor binary_fused_256(const Tensor& A, const Tensor& B, Func op) {
    auto as = A.shape(), bs = B.shape();
    auto os = broadcast_shape(as, bs);
    size_t n = 1;
    for (auto s : os) n *= s;
    Tensor out(os, DType::Float32);
    const float* ap  = get_ptr<float>(A);
    const float* bp  = get_ptr<float>(B);
    float* op_ = get_ptr<float>(out);
    auto om  = build_index_multipliers(os);
    auto ast = shape_to_strides_bytes(as);
    auto bst = shape_to_strides_bytes(bs);
    bool ac = A.is_contiguous() && as == os;
    bool bc = B.is_contiguous() && bs == os;
    bool as_ = A.numel() == 1;
    bool bs_ = B.numel() == 1;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 8) {
        size_t rem = n - i;
        __m256i k = (rem >= 8) ? _mm256_set1_epi32(-1) : tail_mask_ps(rem);
        size_t limit = (rem < 8) ? rem : 8;
        __m256 va, vb;
        
        if      (ac)  va = _mm256_maskload_ps(ap + i, k);
        else if (as_) va = _mm256_set1_ps(ap[0]);
        else {
            int32_t buf[8] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, as, ast);
            va = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ap, _mm256_loadu_si256((__m256i const*)buf), _mm256_castsi256_ps(k), 1);
        }
        
        if      (bc)  vb = _mm256_maskload_ps(bp + i, k);
        else if (bs_) vb = _mm256_set1_ps(bp[0]);
        else {
            int32_t buf[8] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, bs, bst);
            vb = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), bp, _mm256_loadu_si256((__m256i const*)buf), _mm256_castsi256_ps(k), 1);
        }
        _mm256_maskstore_ps(op_ + i, k, op(va, vb));
    }
    return out;
}
 
template<typename Func>
Tensor ternary_fused_256(const Tensor& A, const Tensor& B, const Tensor& C, Func op) {
    auto as = A.shape(), bs = B.shape(), cs = C.shape();
    auto os = broadcast_shape(broadcast_shape(as, bs), cs);
    size_t n = 1;
    for (auto s : os) n *= s;
    Tensor out(os, DType::Float32);
    const float* ap  = get_ptr<float>(A);
    const float* bp  = get_ptr<float>(B);
    const float* cp  = get_ptr<float>(C);
    float* op_ = get_ptr<float>(out);
    auto om  = build_index_multipliers(os);
    auto ast = shape_to_strides_bytes(as);
    auto bst = shape_to_strides_bytes(bs);
    auto cst = shape_to_strides_bytes(cs);
    bool ac = A.is_contiguous() && as == os, as_ = A.numel() == 1;
    bool bc = B.is_contiguous() && bs == os, bs_ = B.numel() == 1;
    bool cc = C.is_contiguous() && cs == os, cs_ = C.numel() == 1;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 8) {
        size_t rem = n - i;
        __m256i k = (rem >= 8) ? _mm256_set1_epi32(-1) : tail_mask_ps(rem);
        size_t limit = (rem < 8) ? rem : 8;
        
        auto load = [&](const float* ptr, bool contig, bool scalar,
                        const std::vector<size_t>& shape,
                        const std::vector<int64_t>& strides) -> __m256 {
            if (contig)  return _mm256_maskload_ps(ptr + i, k);
            if (scalar)  return _mm256_set1_ps(ptr[0]);
            int32_t buf[8] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, shape, strides);
            return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), ptr, _mm256_loadu_si256((__m256i const*)buf), _mm256_castsi256_ps(k), 1);
        };
        __m256 va = load(ap, ac, as_, as, ast);
        __m256 vb = load(bp, bc, bs_, bs, bst);
        __m256 vc = load(cp, cc, cs_, cs, cst);
        _mm256_maskstore_ps(op_ + i, k, op(va, vb, vc));
    }
    return out;
}
 
template<typename Func>
Tensor unary_fused_256(const Tensor& A, Func op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Float32);
    const float* ap  = get_ptr<float>(A);
    float* op_ = get_ptr<float>(out);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 8) {
        size_t rem = n - i;
        __m256i k = (rem >= 8) ? _mm256_set1_epi32(-1) : tail_mask_ps(rem);
        __m256 va = _mm256_maskload_ps(ap + i, k);
        _mm256_maskstore_ps(op_ + i, k, op(va));
    }
    return out;
}

}

Tensor fma_avx2_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256 x, __m256 y, __m256 z) {
        return _mm256_fmadd_ps(x, y, z);
    });
}
 
Tensor fms_avx2_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256 x, __m256 y, __m256 z) {
        return _mm256_fmsub_ps(x, y, z);
    });
}
 
Tensor nfma_avx2_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256 x, __m256 y, __m256 z) {
        return _mm256_fnmadd_ps(x, y, z);
    });
}
 
Tensor add_scale_avx2_f32(const Tensor& a, const Tensor& b, float scale) {
    __m256 vs = _mm256_set1_ps(scale);
    return binary_fused_256(a, b, [vs](__m256 x, __m256 y) {
        return _mm256_mul_ps(_mm256_add_ps(x, y), vs);
    });
}
 
Tensor add_relu_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return relu256_ps(_mm256_add_ps(x, y));
    });
}
 
Tensor add_sigmoid_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return sigmoid256_ps(_mm256_add_ps(x, y));
    });
}
 
Tensor add_tanh_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return tanh256_ps(_mm256_add_ps(x, y));
    });
}
 
Tensor mul_add_avx2_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_avx2_f32(a, b, c);
}
 
 
Tensor add_exp_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return exp256_ps(_mm256_add_ps(x, y));
    });
}
 
Tensor add_ln_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return log256_ps(_mm256_add_ps(x, y));
    });
}
 
Tensor exp_neg_avx2_f32(const Tensor& a) {
    return unary_fused_256(a, [](__m256 x) {
        return exp256_ps(_mm256_xor_ps(x, _mm256_set1_ps(-0.0f)));
    });
}
 
Tensor ln_relu_avx2_f32(const Tensor& a) {
    return unary_fused_256(a, [](__m256 x) {
        return log256_ps(_mm256_max_ps(x, _mm256_setzero_ps()));
    });
}
 
Tensor sigmoid_ln_avx2_f32(const Tensor& a) {
    return unary_fused_256(a, [](__m256 x) {
        return log256_ps(sigmoid256_ps(x));
    });
}
 
Tensor silu_avx2_f32(const Tensor& a) {
    return unary_fused_256(a, [](__m256 x) {
        return silu256_ps(x);
    });
}
 
Tensor gelu_avx2_f32(const Tensor& a) {
    return unary_fused_256(a, [](__m256 x) {
        return gelu256_ps(x);
    });
}
 
Tensor swiglu_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256 x, __m256 y) {
        return _mm256_mul_ps(silu256_ps(x), y);
    });
}
 

Tensor bias_add_relu_avx2_f32(const Tensor& x, const Tensor& bias) {
    return add_relu_avx2_f32(x, bias);
}
 
Tensor bias_add_gelu_avx2_f32(const Tensor& x, const Tensor& bias) {
    return binary_fused_256(x, bias, [](__m256 a, __m256 b) {
        return gelu256_ps(_mm256_add_ps(a, b));
    });
}
 
Tensor scale_shift_avx2_f32(const Tensor& x, float scale, float shift) {
    __m256 vs = _mm256_set1_ps(scale);
    __m256 vb = _mm256_set1_ps(shift);
    return unary_fused_256(x, [vs, vb](__m256 a) {
        return _mm256_fmadd_ps(a, vs, vb);
    });
}
 
Tensor layer_norm_avx2_f32(const Tensor& x, const Tensor& weight,
                              const Tensor& bias, float eps) {
    auto shape = x.shape();
    if (shape.size() < 1) throw std::runtime_error("layer_norm: need at least 1D");
 
    size_t outer = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) outer *= shape[i];
    size_t D = shape.back();
 
    if (weight.numel() != D || bias.numel() != D)
        throw std::runtime_error("layer_norm: weight/bias must match last dim");
 
    Tensor out(shape, DType::Float32);
    const float* xp  = get_ptr<float>(x);
    const float* wp  = get_ptr<float>(weight);
    const float* bp  = get_ptr<float>(bias);
    float* op_ = get_ptr<float>(out);
 
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < outer; ++row) {
        const float* xrow = xp  + row * D;
        float* orow = op_ + row * D;
 
        __m256 vsum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= D; i += 16)
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(xrow + i));
        if (i < D) {
            __m256i k = tail_mask_ps(D - i);
            vsum = _mm256_add_ps(vsum, _mm256_maskload_ps(xrow + i, k));
        }
        float mean = hsum256_ps(vsum) / (float)D;
        __m256 vmean = _mm256_set1_ps(mean);
 
        __m256 vvar = _mm256_setzero_ps();
        i = 0;
        for (; i + 16 <= D; i += 16) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(xrow + i), vmean);
            vvar = _mm256_fmadd_ps(diff, diff, vvar);
        }
        if (i < D) {
            __m256i k = tail_mask_ps(D - i);
            __m256 diff = _mm256_sub_ps(_mm256_maskload_ps(xrow + i, k), vmean);
            __m256 masked_diff = _mm256_and_ps(diff, _mm256_castsi256_ps(k));
            vvar = _mm256_fmadd_ps(diff, masked_diff, vvar);
        }
        float var   = hsum256_ps(vvar) / (float)D;
        float inv_std = 1.0f / std::sqrt(var + eps);
        __m256 vinv = _mm256_set1_ps(inv_std);
 
        i = 0;
        for (; i + 16 <= D; i += 16) {
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xrow + i), vmean), vinv);
            __m256 w    = _mm256_loadu_ps(wp + i);
            __m256 b    = _mm256_loadu_ps(bp + i);
            _mm256_storeu_ps(orow + i, _mm256_fmadd_ps(norm, w, b));
        }
        if (i < D) {
            __m256i k = tail_mask_ps(D - i);
            __m256 norm = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_maskload_ps(xrow + i, k), vmean), vinv);
            __m256 w = _mm256_maskload_ps(wp + i, k);
            __m256 b = _mm256_maskload_ps(bp + i, k);
            _mm256_maskstore_ps(orow + i, k, _mm256_fmadd_ps(norm, w, b));
        }
    }
    return out;
}
 
#endif
