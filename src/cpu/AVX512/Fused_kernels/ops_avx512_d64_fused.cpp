#include "cpu/AVX512/Fused_kernels/ops_avx512_d64_fused.h"
#include "cpu/AVX512/ops_avx512_d64.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>
 
#if defined(__AVX512F__)

namespace{

template<typename T>
inline T* get_ptr(const Tensor& t){
    return (T*)t.impl->data->data.get() +t.impl->offset;
}

#define ZMM_0_PD _mm512_setzero_pd()
#define ZMM_1_PD _mm512_set1_pd(1.0)
#define ZMM_05_PD _mm512_set1_pd(0.5)

inline __mmask8 tail_mask_d64(size_t n) { 
    return (__mmask8)((1U << n) - 1); 
}

inline __m512d bitwise_xor(__m512d a, __m512d b) {
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
inline __m512d bitwise_and(__m512d a, __m512d b) {
    return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
inline __m512d bitwise_or(__m512d a, __m512d b) {
    return _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
inline __m512d abs_pd(__m512d x) {
    return _mm512_castsi512_pd(_mm512_and_si512(
        _mm512_castpd_si512(x), 
        _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFull)
    ));
}

inline __m512d exp512_pd(__m512d x) {
    __m512d one = ZMM_1_PD;
    x  = _mm512_min_pd(x, _mm512_set1_pd( 709.43702074));
    x  = _mm512_max_pd(x, _mm512_set1_pd(-709.43702074));
    __m512d fx = _mm512_fmadd_pd(x, _mm512_set1_pd(1.44269504088896340736), ZMM_05_PD);
    fx = _mm512_floor_pd(fx);
    __m512d tmp = _mm512_mul_pd(fx, _mm512_set1_pd(6.93147180369123816490e-1));
    __m512d z   = _mm512_mul_pd(fx, _mm512_set1_pd(1.90821492927058770002e-10));
    x  = _mm512_sub_pd(x, tmp);
    x  = _mm512_sub_pd(x, z);
    __m512d y = _mm512_set1_pd(2.08860621107283687536341e-9);
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(2.51112930892876518610684e-8));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(2.75573162723986771042917e-7));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(2.75573905348516390806847e-6));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(2.48015872867325702009358e-5));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(1.98412698411804183580850e-4));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(1.38888888888679612361289e-3));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(8.33333333333329931873939e-3));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(4.16666666666666657414808e-2));
    y = _mm512_fmadd_pd(y, x, _mm512_set1_pd(1.66666666666666685227835e-1));
    y = _mm512_fmadd_pd(y, x, ZMM_05_PD);
    z = _mm512_mul_pd(x, x);
    y = _mm512_fmadd_pd(y, z, x);
    y = _mm512_add_pd(y, one);
    __m512i emm0 = _mm512_cvttpd_epi64(fx);           // AVX512DQ
    emm0 = _mm512_add_epi64(emm0, _mm512_set1_epi64(0x3ff));
    emm0 = _mm512_slli_epi64(emm0, 52);
    return _mm512_mul_pd(y, _mm512_castsi512_pd(emm0));
}

inline __m512d log512_pd(__m512d x) {
    x = _mm512_max_pd(x, _mm512_set1_pd(1e-300));
    __m512i xi    = _mm512_castpd_si512(x);
    __m512i exp_i = _mm512_srli_epi64(xi, 52);
    exp_i         = _mm512_sub_epi64(exp_i, _mm512_set1_epi64(1023));
    __m512d e     = _mm512_cvtepi64_pd(exp_i);        
    xi = _mm512_and_si512(xi, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL));
    xi = _mm512_or_si512 (xi, _mm512_set1_epi64(0x3FF0000000000000LL));
    x  = _mm512_castsi512_pd(xi);
    __mmask8 mask = _mm512_cmp_pd_mask(x, _mm512_set1_pd(1.41421356237309504880), _CMP_GT_OS);
    x = _mm512_mask_mul_pd(x, mask, x, _mm512_set1_pd(0.5));
    e = _mm512_mask_add_pd(e, mask, e, ZMM_1_PD);
    __m512d r  = _mm512_div_pd(_mm512_sub_pd(x, ZMM_1_PD), _mm512_add_pd(x, ZMM_1_PD));
    __m512d r2 = _mm512_mul_pd(r, r);
    __m512d p = _mm512_set1_pd(7.33333333333333593622e-2);
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(7.69230769230769468571e-2));
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(9.09090909090906940187e-2));
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(1.11111111111111110869e-1));
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(1.42857142857142872628e-1));
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(1.99999999999999994671e-1));
    p = _mm512_fmadd_pd(p, r2, _mm512_set1_pd(3.33333333333333331483e-1));
    p = _mm512_mul_pd(p, r2);
    p = _mm512_fmadd_pd(p, r, r);
    __m512d ln_m = _mm512_add_pd(p, p);
    return _mm512_fmadd_pd(e, _mm512_set1_pd(6.93147180559945309417e-1), ln_m);
}

inline __m512d sigmoid512_pd(__m512d x) {
    __m512d neg_x = bitwise_xor(x, _mm512_set1_pd(-0.0));
    return _mm512_div_pd(ZMM_1_PD, _mm512_add_pd(ZMM_1_PD, exp512_pd(neg_x)));
}
 
inline __m512d tanh512_pd(__m512d x) {
    __m512d two_x  = _mm512_mul_pd(x, _mm512_set1_pd(2.0));
    __m512d exp_2x = exp512_pd(two_x);
    return _mm512_div_pd(_mm512_sub_pd(exp_2x, ZMM_1_PD),
                         _mm512_add_pd(exp_2x, ZMM_1_PD));
}
 
inline __m512d relu512_pd(__m512d x) { return _mm512_max_pd(x, ZMM_0_PD); }
 
inline __m512d silu512_pd(__m512d x) { return _mm512_mul_pd(x, sigmoid512_pd(x)); }
 
inline __m512d gelu512_pd(__m512d x) {
    return _mm512_mul_pd(x, sigmoid512_pd(_mm512_mul_pd(x, _mm512_set1_pd(1.702))));
}
 
inline double hsum512_pd(__m512d v) {
    __m256d lo  = _mm512_castpd512_pd256(v);
    __m256d hi  = _mm512_extractf64x4_pd(v, 1);
    __m256d sum = _mm256_add_pd(lo, hi);
    __m128d a   = _mm256_castpd256_pd128(sum);
    __m128d b   = _mm256_extractf128_pd(sum, 1);
    a = _mm_add_pd(a, b);
    __m128d shuf = _mm_unpackhi_pd(a, a);
    return _mm_cvtsd_f64(_mm_add_pd(a, shuf));
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
    s.back() = sizeof(double);
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
    Tensor out(os, DType::Double64);
    const double* ap  = get_ptr<double>(A);
    const double* bp  = get_ptr<double>(B);
    double* op_ = get_ptr<double>(out);
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
        __mmask8 k = (rem >= 8) ? 0xFF : tail_mask_d64(rem);
        __m512d va, vb;
        if      (ac)  va = _mm512_maskz_loadu_pd(k, ap + i);
        else if (as_) va = _mm512_set1_pd(ap[0]);
        else {
            int32_t buf[8] = {};
            for (int l = 0; l < 8 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, as, ast);
            va = _mm512_mask_i32gather_pd(ZMM_0_PD, k, _mm256_loadu_si256((__m256i const*)buf), ap, 1);
        }
        if      (bc)  vb = _mm512_maskz_loadu_pd(k, bp + i);
        else if (bs_) vb = _mm512_set1_pd(bp[0]);
        else {
            int32_t buf[8] = {};
            for (int l = 0; l < 8 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, bs, bst);
            vb = _mm512_mask_i32gather_pd(ZMM_0_PD, k, _mm256_loadu_si256((__m256i const*)buf), bp, 1);
        }
        _mm512_mask_storeu_pd(op_ + i, k, op(va, vb));
    }
    return out;
}
 
template<typename Func>
Tensor ternary_fused_512(const Tensor& A, const Tensor& B, const Tensor& C, Func op) {
    auto as = A.shape(), bs = B.shape(), cs = C.shape();
    auto os = broadcast_shape(broadcast_shape(as, bs), cs);
    size_t n = 1;
    for (auto s : os) n *= s;
    Tensor out(os, DType::Double64);
    const double* ap  = get_ptr<double>(A);
    const double* bp  = get_ptr<double>(B);
    const double* cp  = get_ptr<double>(C);
    double* op_ = get_ptr<double>(out);
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
        __mmask8 k = (rem >= 8) ? 0xFF : tail_mask_d64(rem);
        auto load = [&](const double* ptr, bool contig, bool scalar,
                        const std::vector<size_t>& shape,
                        const std::vector<int64_t>& strides) -> __m512d {
            if (contig)  return _mm512_maskz_loadu_pd(k, ptr + i);
            if (scalar)  return _mm512_set1_pd(ptr[0]);
            int32_t buf[8] = {};
            for (int l = 0; l < 8 && (k >> l & 1); ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, shape, strides);
            return _mm512_mask_i32gather_pd(ZMM_0_PD, k, _mm256_loadu_si256((__m256i const*)buf), ptr, 1);
        };
        __m512d va = load(ap, ac, as_, as, ast);
        __m512d vb = load(bp, bc, bs_, bs, bst);
        __m512d vc = load(cp, cc, cs_, cs, cst);
        _mm512_mask_storeu_pd(op_ + i, k, op(va, vb, vc));
    }
    return out;
}
 
template<typename Func>
Tensor unary_fused_512(const Tensor& A, Func op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Double64);
    const double* ap  = get_ptr<double>(A);
    double* op_ = get_ptr<double>(out);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 8) {
        size_t rem = n - i;
        __mmask8 k = (rem >= 8) ? 0xFF : tail_mask_d64(rem);
        __m512d va = _mm512_maskz_loadu_pd(k, ap + i);
        _mm512_mask_storeu_pd(op_ + i, k, op(va));
    }
    return out;
}

}

Tensor fma_avx512_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512d x, __m512d y, __m512d z) {
        return _mm512_fmadd_pd(x, y, z);
    });
}
 
Tensor fms_avx512_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512d x, __m512d y, __m512d z) {
        return _mm512_fmsub_pd(x, y, z);
    });
}
 
Tensor nfma_avx512_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_512(a, b, c, [](__m512d x, __m512d y, __m512d z) {
        return _mm512_fnmadd_pd(x, y, z);
    });
}
 
Tensor add_scale_avx512_d64(const Tensor& a, const Tensor& b, float scale) {
    __m512d vs = _mm512_set1_pd((double)scale);
    return binary_fused_512(a, b, [vs](__m512d x, __m512d y) {
        return _mm512_mul_pd(_mm512_add_pd(x, y), vs);
    });
}
 
Tensor add_relu_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return relu512_pd(_mm512_add_pd(x, y));
    });
}
 
Tensor add_sigmoid_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return sigmoid512_pd(_mm512_add_pd(x, y));
    });
}
 
Tensor add_tanh_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return tanh512_pd(_mm512_add_pd(x, y));
    });
}
 
Tensor mul_add_avx512_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_avx512_d64(a, b, c);
}
 
Tensor add_exp_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return exp512_pd(_mm512_add_pd(x, y));
    });
}
 
Tensor add_ln_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return log512_pd(_mm512_add_pd(x, y));
    });
}
 
Tensor exp_neg_avx512_d64(const Tensor& a) {
    return unary_fused_512(a, [](__m512d x) {
        return exp512_pd(bitwise_xor(x, _mm512_set1_pd(-0.0)));
    });
}
 
Tensor ln_relu_avx512_d64(const Tensor& a) {
    return unary_fused_512(a, [](__m512d x) {
        return log512_pd(_mm512_max_pd(x, ZMM_0_PD));
    });
}
 
Tensor sigmoid_ln_avx512_d64(const Tensor& a) {
    return unary_fused_512(a, [](__m512d x) {
        return log512_pd(sigmoid512_pd(x));
    });
}
 
Tensor silu_avx512_d64(const Tensor& a) {
    return unary_fused_512(a, [](__m512d x) {
        return silu512_pd(x);
    });
}
 
Tensor gelu_avx512_d64(const Tensor& a) {
    return unary_fused_512(a, [](__m512d x) {
        return gelu512_pd(x);
    });
}
 
Tensor swiglu_avx512_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_512(a, b, [](__m512d x, __m512d y) {
        return _mm512_mul_pd(silu512_pd(x), y);
    });
}

Tensor bias_add_relu_avx512_d64(const Tensor& x, const Tensor& bias) {
    return add_relu_avx512_d64(x, bias);
}
 
Tensor bias_add_gelu_avx512_d64(const Tensor& x, const Tensor& bias) {
    return binary_fused_512(x, bias, [](__m512d a, __m512d b) {
        return gelu512_pd(_mm512_add_pd(a, b));
    });
}
 
Tensor scale_shift_avx512_d64(const Tensor& x, float scale, float shift) {
    __m512d vs = _mm512_set1_pd((double)scale);
    __m512d vb = _mm512_set1_pd((double)shift);
    return unary_fused_512(x, [vs, vb](__m512d a) {
        return _mm512_fmadd_pd(a, vs, vb);
    });
}
 
Tensor layer_norm_avx512_d64(const Tensor& x, const Tensor& weight,
                              const Tensor& bias, float eps) {
    auto shape = x.shape();
    if (shape.size() < 1) throw std::runtime_error("layer_norm: need at least 1D");
 
    size_t outer = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) outer *= shape[i];
    size_t D = shape.back();
 
    if (weight.numel() != D || bias.numel() != D)
        throw std::runtime_error("layer_norm: weight/bias must match last dim");
 
    Tensor out(shape, DType::Double64);
    const double* xp  = get_ptr<double>(x);
    const double* wp  = get_ptr<double>(weight);
    const double* bp  = get_ptr<double>(bias);
    double* op_ = get_ptr<double>(out);
 
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < outer; ++row) {
        const double* xrow = xp  + row * D;
        double* orow = op_ + row * D;
 
        __m512d vsum = ZMM_0_PD;
        size_t i = 0;
        for (; i + 8 <= D; i += 8)
            vsum = _mm512_add_pd(vsum, _mm512_loadu_pd(xrow + i));
        if (i < D) {
            __mmask8 k = tail_mask_d64(D - i);
            vsum = _mm512_add_pd(vsum, _mm512_maskz_loadu_pd(k, xrow + i));
        }
        double mean = hsum512_pd(vsum) / (double)D;
        __m512d vmean = _mm512_set1_pd(mean);
 
        __m512d vvar = ZMM_0_PD;
        i = 0;
        for (; i + 8 <= D; i += 8) {
            __m512d diff = _mm512_sub_pd(_mm512_loadu_pd(xrow + i), vmean);
            vvar = _mm512_fmadd_pd(diff, diff, vvar);
        }
        if (i < D) {
            __mmask8 k = tail_mask_d64(D - i);
            __m512d diff = _mm512_sub_pd(_mm512_maskz_loadu_pd(k, xrow + i), vmean);
            vvar = _mm512_fmadd_pd(diff, _mm512_maskz_mov_pd(k, diff), vvar);
        }
        double var   = hsum512_pd(vvar) / (double)D;
        double inv_std = 1.0 / std::sqrt(var + (double)eps);
        __m512d vinv = _mm512_set1_pd(inv_std);
 
        i = 0;
        for (; i + 8 <= D; i += 8) {
            __m512d norm = _mm512_mul_pd(_mm512_sub_pd(_mm512_loadu_pd(xrow + i), vmean), vinv);
            __m512d w    = _mm512_loadu_pd(wp + i);
            __m512d b    = _mm512_loadu_pd(bp + i);
            _mm512_storeu_pd(orow + i, _mm512_fmadd_pd(norm, w, b));
        }
        if (i < D) {
            __mmask8 k = tail_mask_d64(D - i);
            __m512d norm = _mm512_mul_pd(
                _mm512_sub_pd(_mm512_maskz_loadu_pd(k, xrow + i), vmean), vinv);
            __m512d w = _mm512_maskz_loadu_pd(k, wp + i);
            __m512d b = _mm512_maskz_loadu_pd(k, bp + i);
            _mm512_mask_storeu_pd(orow + i, k, _mm512_fmadd_pd(norm, w, b));
        }
    }
    return out;
}
 
#endif