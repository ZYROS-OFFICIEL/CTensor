#include "ops_avx512_d64.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>

#if defined(__AVX512F__)

namespace {
//                     Internal AVX-512 Constants & Helpers

const __m512d _zmm_1  = _mm512_set1_pd(1.0);
const __m512d _zmm_0  = _mm512_setzero_pd();
const __m512d _zmm_nan= _mm512_set1_pd(NAN);

// Helper to generate a mask for the first 'n' elements (0 <= n <= 8)
inline __mmask8 tail_mask(size_t n) {
    return (__mmask8)((1U << n) - 1);
}

// --- Horizontal Sum for ZMM (Double) ---
inline double hsum512_pd(__m512d v) {
    // reduce 512 -> 256
    __m256d vlow = _mm512_castpd512_pd256(v);
    __m256d vhigh = _mm512_extractf64x4_pd(v, 1);
    vlow = _mm256_add_pd(vlow, vhigh);
    // reduce 256 -> 128
    __m128d xlow = _mm256_castpd256_pd128(vlow);
    __m128d xhigh = _mm256_extractf128_pd(vlow, 1);
    xlow = _mm_add_pd(xlow, xhigh);
    // reduce 128 -> 64
    __m128d shuf = _mm_movedup_pd(xlow); // broadcast high to low
    __m128d sums = _mm_add_pd(xlow, shuf);
    return _mm_cvtsd_f64(sums);
}

} // namespace