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

}

#endif