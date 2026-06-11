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
}