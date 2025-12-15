#include "ops_avx512_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>

#if defined(__AVX512F__)

namespace {

// ========================================================================
//                     Internal AVX-512 Math Constants & Helpers
// ========================================================================

const __m512 _zmm_1  = _mm512_set1_ps(1.0f);
const __m512 _zmm_05 = _mm512_set1_ps(0.5f);
const __m512 _zmm_0  = _mm512_setzero_ps();
const __m512 _zmm_nan= _mm512_set1_ps(NAN);

// Helper to generate a mask for the first 'n' elements (0 <= n <= 16)
inline __mmask16 tail_mask(size_t n) {
    return (__mmask16)((1U << n) - 1);
}

// --- Abs ---
inline __m512 _mm512_abs_ps(__m512 x) {
    // Clear sign bit using integer casting (requires AVX512DQ usually, but we can do AND with mask)
    // AVX512F has bitwise ops on registers.
    // 0x7FFFFFFF mask
    __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), mask));
}

// --- Exponential (Exp) for AVX-512 ---
// Ported Cephes approximation for ZMM
inline __m512 exp512_ps(__m512 x) {
    __m512 fx, one = _zmm_1;
    
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));

    fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), _zmm_05);
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
    y = _mm512_fmadd_ps(y, x, _zmm_05);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    // Build 2^n
    __m512i emm0 = _mm512_cvttps_epi32(fx);
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    emm0 = _mm512_slli_epi32(emm0, 23);
    
    return _mm512_mul_ps(y, _mm512_castsi512_ps(emm0));
}

// --- Logarithm (Ln) for AVX-512 ---
inline __m512 log512_ps(__m512 x) {
    __m512 one = _zmm_1;
    __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, _zmm_0, _CMP_LE_OQ);
    
    x = _mm512_max_ps(x, _mm512_set1_ps(1.17549435e-38f));

    __m512i emm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
    
    // keep mantissa
    x = _mm512_and_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffff)));
    x = _mm512_or_ps(x, _zmm_05);

    emm0 = _mm512_sub_epi32(emm0, _mm512_set1_epi32(0x7f));
    __m512 e = _mm512_cvtepi32_ps(emm0);
    e = _mm512_add_ps(e, one);

    __mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    __m512 tmp = _mm512_mask_z_mov_ps(mask, x); // zero out where not mask if needed, but logic below handles it
    
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
    y = _mm512_fnmadd_ps(z, _zmm_05, y); // y = y - z*0.5
    
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, _mm512_set1_ps(0.693359375f), x);

    // NaNs
    return _mm512_mask_blend_ps(invalid_mask, x, _zmm_nan); 
}
