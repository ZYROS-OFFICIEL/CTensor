#include "ops_avx2_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>

#if defined(__AVX2__)

// ========================================================================
//                     Internal AVX2 Math Library
// ========================================================================
// High-performance polynomial approximations for transcendental functions
// Adapted for single-file, self-contained AVX2 usage.

namespace {

// --- Constants ---
const __m256 _ps_1  = _mm256_set1_ps(1.0f);
const __m256 _ps_05 = _mm256_set1_ps(0.5f);
const __m256 _ps_0  = _mm256_setzero_ps();
const __m256 _ps_nan= _mm256_set1_ps(NAN);
const __m256i _pi32_0x7f = _mm256_set1_epi32(0x7f);

// --- Helpers ---
inline __m256 _mm256_abs_ps(__m256 x) {
    static const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask));
}

inline __m256 _mm256_neg_ps(__m256 x) {
    return _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
}

// --- Exponential (Exp) ---
// Cephes-style approximation
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
