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
