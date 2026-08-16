#include "cpu/AVX2/Fused_kernels/ops_avx2_d64_fused.h"
#include "cpu/AVX2/ops_avx2_d64.h"
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>
 
#if defined(__AVX2__)

namespace{

template<typename T>
inline T* get_ptr(const Tensor& t){
    return (T*)t.impl->data->data.get() +t.impl->offset;
}

inline __m256i tail_mask_pd(size_t rem) {
    static const int64_t mask_table[8] = {
        -1, -1, -1, -1,
         0,  0,  0,  0
    };
    return _mm256_loadu_si256((const __m256i*)&mask_table[4 - rem]);
}

inline __m256d abs_pd(__m256d x) {
    return _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)));
}

inline __m256d exp256_pd(__m256d x) {
    __m256d one = _mm256_set1_pd(1.0);
    x  = _mm256_min_pd(x, _mm256_set1_pd( 709.43702074));
    x  = _mm256_max_pd(x, _mm256_set1_pd(-709.43702074));
    __m256d fx = _mm256_fmadd_pd(x, _mm256_set1_pd(1.44269504088896340736), _mm256_set1_pd(0.5));
    fx = _mm256_floor_pd(fx);
    __m256d tmp = _mm256_mul_pd(fx, _mm256_set1_pd(6.93147180369123816490e-1));
    __m256d z   = _mm256_mul_pd(fx, _mm256_set1_pd(1.90821492927058770002e-10));
    x  = _mm256_sub_pd(x, tmp);
    x  = _mm256_sub_pd(x, z);
    __m256d y = _mm256_set1_pd(2.08860621107283687536341e-9);
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(2.51112930892876518610684e-8));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(2.75573162723986771042917e-7));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(2.75573905348516390806847e-6));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(2.48015872867325702009358e-5));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(1.98412698411804183580850e-4));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(1.38888888888679612361289e-3));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(8.33333333333329931873939e-3));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(4.16666666666666657414808e-2));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(1.66666666666666685227835e-1));
    y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(0.5));
    z = _mm256_mul_pd(x, x);
    y = _mm256_fmadd_pd(y, z, x);
    y = _mm256_add_pd(y, one);
    
    // AVX2 translation of 64-bit integer conversion
    __m128i emm0_32 = _mm256_cvttpd_epi32(fx);
    __m256i emm0 = _mm256_cvtepi32_epi64(emm0_32);
    emm0 = _mm256_add_epi64(emm0, _mm256_set1_epi64x(0x3ff));
    emm0 = _mm256_slli_epi64(emm0, 52);
    return _mm256_mul_pd(y, _mm256_castsi256_pd(emm0));
}

inline __m256d log256_pd(__m256d x) {
    x = _mm256_max_pd(x, _mm256_set1_pd(1e-300));
    __m256i xi    = _mm256_castpd_si256(x);
    __m256i exp_i = _mm256_srli_epi64(xi, 52);
    exp_i         = _mm256_sub_epi64(exp_i, _mm256_set1_epi64x(1023));
    
    // Pack 64-bit ints to 32-bit ints safely so we can convert to double in AVX2
    __m256i x1 = _mm256_shuffle_epi32(exp_i, _MM_SHUFFLE(2, 0, 2, 0));
    __m128i low = _mm256_castsi256_si128(x1);
    __m128i high = _mm256_extracti128_si256(x1, 1);
    __m128i packed = _mm_unpacklo_epi64(low, high);
    __m256d e = _mm256_cvtepi32_pd(packed);
    
    xi = _mm256_and_si256(xi, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL));
    xi = _mm256_or_si256 (xi, _mm256_set1_epi64x(0x3FF0000000000000LL));
    x  = _mm256_castsi256_pd(xi);
    
    __m256d mask = _mm256_cmp_pd(x, _mm256_set1_pd(1.41421356237309504880), _CMP_GT_OS);
    x = _mm256_blendv_pd(x, _mm256_mul_pd(x, _mm256_set1_pd(0.5)), mask);
    e = _mm256_blendv_pd(e, _mm256_add_pd(e, _mm256_set1_pd(1.0)), mask);
    
    __m256d r  = _mm256_div_pd(_mm256_sub_pd(x, _mm256_set1_pd(1.0)), _mm256_add_pd(x, _mm256_set1_pd(1.0)));
    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d p = _mm256_set1_pd(7.33333333333333593622e-2);
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(7.69230769230769468571e-2));
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(9.09090909090906940187e-2));
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(1.11111111111111110869e-1));
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(1.42857142857142872628e-1));
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(1.99999999999999994671e-1));
    p = _mm256_fmadd_pd(p, r2, _mm256_set1_pd(3.33333333333333331483e-1));
    p = _mm256_mul_pd(p, r2);
    p = _mm256_fmadd_pd(p, r, r);
    __m256d ln_m = _mm256_add_pd(p, p);
    return _mm256_fmadd_pd(e, _mm256_set1_pd(6.93147180559945309417e-1), ln_m);
}

inline __m256d sigmoid256_pd(__m256d x) {
    __m256d neg_x = _mm256_xor_pd(x, _mm256_set1_pd(-0.0));
    return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_add_pd(_mm256_set1_pd(1.0), exp256_pd(neg_x)));
}
 
inline __m256d tanh256_pd(__m256d x) {
    __m256d two_x  = _mm256_mul_pd(x, _mm256_set1_pd(2.0));
    __m256d exp_2x = exp256_pd(two_x);
    return _mm256_div_pd(_mm256_sub_pd(exp_2x, _mm256_set1_pd(1.0)),
                         _mm256_add_pd(exp_2x, _mm256_set1_pd(1.0)));
}
 
inline __m256d relu256_pd(__m256d x) { return _mm256_max_pd(x, _mm256_setzero_pd()); }
 
inline __m256d silu256_pd(__m256d x) { return _mm256_mul_pd(x, sigmoid256_pd(x)); }
 
inline __m256d gelu256_pd(__m256d x) {
    return _mm256_mul_pd(x, sigmoid256_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.702))));
}
 
inline double hsum256_pd(__m256d v) {
    __m128d a = _mm256_extractf128_pd(v, 1);
    __m128d b = _mm256_castpd256_pd128(v);
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
Tensor binary_fused_256(const Tensor& A, const Tensor& B, Func op) {
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
    for (size_t i = 0; i < n; i += 4) {
        size_t rem = n - i;
        __m256i k = (rem >= 4) ? _mm256_set1_epi64x(-1) : tail_mask_pd(rem);
        size_t limit = (rem < 4) ? rem : 4;
        __m256d va, vb;
        
        if      (ac)  va = _mm256_maskload_pd(ap + i, k);
        else if (as_) va = _mm256_set1_pd(ap[0]);
        else {
            int32_t buf[4] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, as, ast);
            va = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), ap, _mm_loadu_si128((__m128i const*)buf), _mm256_castsi256_pd(k), 1);
        }
        
        if      (bc)  vb = _mm256_maskload_pd(bp + i, k);
        else if (bs_) vb = _mm256_set1_pd(bp[0]);
        else {
            int32_t buf[4] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, bs, bst);
            vb = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), bp, _mm_loadu_si128((__m128i const*)buf), _mm256_castsi256_pd(k), 1);
        }
        _mm256_maskstore_pd(op_ + i, k, op(va, vb));
    }
    return out;
}
 
template<typename Func>
Tensor ternary_fused_256(const Tensor& A, const Tensor& B, const Tensor& C, Func op) {
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
    for (size_t i = 0; i < n; i += 4) {
        size_t rem = n - i;
        __m256i k = (rem >= 4) ? _mm256_set1_epi64x(-1) : tail_mask_pd(rem);
        size_t limit = (rem < 4) ? rem : 4;
        
        auto load = [&](const double* ptr, bool contig, bool scalar,
                        const std::vector<size_t>& shape,
                        const std::vector<int64_t>& strides) -> __m256d {
            if (contig)  return _mm256_maskload_pd(ptr + i, k);
            if (scalar)  return _mm256_set1_pd(ptr[0]);
            int32_t buf[4] = {};
            for (size_t l = 0; l < limit; ++l)
                buf[l] = compute_offset_bytes(i + l, os, om, shape, strides);
            return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), ptr, _mm_loadu_si128((__m128i const*)buf), _mm256_castsi256_pd(k), 1);
        };
        __m256d va = load(ap, ac, as_, as, ast);
        __m256d vb = load(bp, bc, bs_, bs, bst);
        __m256d vc = load(cp, cc, cs_, cs, cst);
        _mm256_maskstore_pd(op_ + i, k, op(va, vb, vc));
    }
    return out;
}
 
template<typename Func>
Tensor unary_fused_256(const Tensor& A, Func op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Double64);
    const double* ap  = get_ptr<double>(A);
    double* op_ = get_ptr<double>(out);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 4) {
        size_t rem = n - i;
        __m256i k = (rem >= 4) ? _mm256_set1_epi64x(-1) : tail_mask_pd(rem);
        __m256d va = _mm256_maskload_pd(ap + i, k);
        _mm256_maskstore_pd(op_ + i, k, op(va));
    }
    return out;
}

}

Tensor fma_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256d x, __m256d y, __m256d z) {
        return _mm256_fmadd_pd(x, y, z);
    });
}
 
Tensor fms_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256d x, __m256d y, __m256d z) {
        return _mm256_fmsub_pd(x, y, z);
    });
}
 
Tensor nfma_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return ternary_fused_256(a, b, c, [](__m256d x, __m256d y, __m256d z) {
        return _mm256_fnmadd_pd(x, y, z);
    });
}
 
Tensor add_scale_avx2_d64(const Tensor& a, const Tensor& b, float scale) {
    __m256d vs = _mm256_set1_pd((double)scale);
    return binary_fused_256(a, b, [vs](__m256d x, __m256d y) {
        return _mm256_mul_pd(_mm256_add_pd(x, y), vs);
    });
}
 
Tensor add_relu_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return relu256_pd(_mm256_add_pd(x, y));
    });
}
 
Tensor add_sigmoid_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return sigmoid256_pd(_mm256_add_pd(x, y));
    });
}
 
Tensor add_tanh_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return tanh256_pd(_mm256_add_pd(x, y));
    });
}
 
Tensor mul_add_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_avx2_d64(a, b, c);
}
 
Tensor add_exp_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return exp256_pd(_mm256_add_pd(x, y));
    });
}
 
Tensor add_ln_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return log256_pd(_mm256_add_pd(x, y));
    });
}
 
Tensor exp_neg_avx2_d64(const Tensor& a) {
    return unary_fused_256(a, [](__m256d x) {
        return exp256_pd(_mm256_xor_pd(x, _mm256_set1_pd(-0.0)));
    });
}
 
Tensor ln_relu_avx2_d64(const Tensor& a) {
    return unary_fused_256(a, [](__m256d x) {
        return log256_pd(_mm256_max_pd(x, _mm256_setzero_pd()));
    });
}
 
Tensor sigmoid_ln_avx2_d64(const Tensor& a) {
    return unary_fused_256(a, [](__m256d x) {
        return log256_pd(sigmoid256_pd(x));
    });
}
 
Tensor swiglu_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_fused_256(a, b, [](__m256d x, __m256d y) {
        return _mm256_mul_pd(silu256_pd(x), y);
    });
}

Tensor bias_add_relu_avx2_d64(const Tensor& x, const Tensor& bias) {
    return add_relu_avx2_d64(x, bias);
}
 
Tensor bias_add_gelu_avx2_d64(const Tensor& x, const Tensor& bias) {
    return binary_fused_256(x, bias, [](__m256d a, __m256d b) {
        return gelu256_pd(_mm256_add_pd(a, b));
    });
}
 
Tensor scale_shift_avx2_d64(const Tensor& x, float scale, float shift) {
    __m256d vs = _mm256_set1_pd((double)scale);
    __m256d vb = _mm256_set1_pd((double)shift);
    return unary_fused_256(x, [vs, vb](__m256d a) {
        return _mm256_fmadd_pd(a, vs, vb);
    });
}
 
Tensor layer_norm_avx2_d64(const Tensor& x, const Tensor& weight,
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
 
        __m256d vsum = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 4 <= D; i += 4)
            vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(xrow + i));
        if (i < D) {
            __m256i k = tail_mask_pd(D - i);
            vsum = _mm256_add_pd(vsum, _mm256_maskload_pd(xrow + i, k));
        }
        double mean = hsum256_pd(vsum) / (double)D;
        __m256d vmean = _mm256_set1_pd(mean);
 
        __m256d vvar = _mm256_setzero_pd();
        i = 0;
        for (; i + 4 <= D; i += 4) {
            __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(xrow + i), vmean);
            vvar = _mm256_fmadd_pd(diff, diff, vvar);
        }
        if (i < D) {
            __m256i k = tail_mask_pd(D - i);
            __m256d diff = _mm256_sub_pd(_mm256_maskload_pd(xrow + i, k), vmean);
            __m256d masked_diff = _mm256_and_pd(diff, _mm256_castsi256_pd(k));
            vvar = _mm256_fmadd_pd(diff, masked_diff, vvar);
        }
        double var   = hsum256_pd(vvar) / (double)D;
        double inv_std = 1.0 / std::sqrt(var + (double)eps);
        __m256d vinv = _mm256_set1_pd(inv_std);
 
        i = 0;
        for (; i + 4 <= D; i += 4) {
            __m256d norm = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(xrow + i), vmean), vinv);
            __m256d w    = _mm256_loadu_pd(wp + i);
            __m256d b    = _mm256_loadu_pd(bp + i);
            _mm256_storeu_pd(orow + i, _mm256_fmadd_pd(norm, w, b));
        }
        if (i < D) {
            __m256i k = tail_mask_pd(D - i);
            __m256d norm = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_maskload_pd(xrow + i, k), vmean), vinv);
            __m256d w = _mm256_maskload_pd(wp + i, k);
            __m256d b = _mm256_maskload_pd(bp + i, k);
            _mm256_maskstore_pd(orow + i, k, _mm256_fmadd_pd(norm, w, b));
        }
    }
    return out;
}
 
#endif