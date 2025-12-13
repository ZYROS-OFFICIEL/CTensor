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

// --- Natural Logarithm (Ln) ---
inline __m256 log256_ps(__m256 x) {
    __m256 one = _ps_1;
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);

    x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f)); // avoid denormal/zero

    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    
    // keep only the fractional part
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
    
    x = _mm256_or_ps(x, invalid_mask); // propagate NaNs
    return x;
}

// --- Sine (Sin) ---
inline __m256 sin256_ps(__m256 x) {
    __m256 xmm1, xmm2, xmm3, sign_bit, y;
    __m256i emm0, emm2;
    sign_bit = x;
    x = _mm256_abs_ps(x);

    xmm1 = _mm256_mul_ps(x, _mm256_set1_ps(0.63661977236758134308f)); // 2/pi
    emm2 = _mm256_cvttps_epi32(xmm1);
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(emm2, _mm256_set1_epi32(4)), _mm256_setzero_si256()));
    sign_bit = _mm256_xor_ps(sign_bit, _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), poly_mask)); // swap sign if needed

    __m256 m1 = _mm256_mul_ps(y, _mm256_set1_ps(-1.5703125f)); // -PI/2 1
    __m256 m2 = _mm256_mul_ps(y, _mm256_set1_ps(-4.837512969970703125e-4f)); // -PI/2 2
    __m256 m3 = _mm256_mul_ps(y, _mm256_set1_ps(-7.549789948768648e-8f)); // -PI/2 3
    
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
    // cos(x) = sin(x + pi/2)
    x = _mm256_add_ps(x, _mm256_set1_ps(1.57079632679489661923f));
    return sin256_ps(x);
}

// --- Tanh ---
inline __m256 tanh256_ps(__m256 x) {
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    __m256 two_x = _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    __m256 exp_2x = exp256_ps(two_x);
    __m256 num = _mm256_sub_ps(exp_2x, _ps_1);
    __m256 den = _mm256_add_ps(exp_2x, _ps_1);
    return _mm256_div_ps(num, den);
}

// --- Sigmoid ---
inline __m256 sigmoid256_ps(__m256 x) {
    // 1 / (1 + exp(-x))
    __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    __m256 e = exp256_ps(neg_x);
    __m256 den = _mm256_add_ps(_ps_1, e);
    return _mm256_div_ps(_ps_1, den);
}
// --- Pow ---
inline __m256 pow256_ps(__m256 a, __m256 b) {
    // a^b = exp(b * ln(a))
    // Note: this implementation propagates NaNs if a <= 0
    return exp256_ps(_mm256_mul_ps(b, log256_ps(a)));
}