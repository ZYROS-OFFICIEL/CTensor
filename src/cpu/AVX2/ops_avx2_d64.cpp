#include "ops_avx2_d64.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>

#if defined(__AVX2__)

namespace {

// --- Constants (Double) ---
const __m256d _pd_1  = _mm256_set1_pd(1.0);
const __m256d _pd_0  = _mm256_setzero_pd();

// --- Helpers ---

inline void build_tail_mask_d64(int64_t* mask_buffer, size_t n_remaining) {
    // maskload_pd requires 64-bit integer mask. High bit 1 = keep, 0 = discard.
    for (size_t i = 0; i < 4; ++i) {
        mask_buffer[i] = (i < n_remaining) ? -1LL : 0LL; // -1LL is all ones, 0 is all zeros
    }
}

inline __m256d masked_loadu_pd(const double* ptr, const int64_t* mask) {
    __m256i vmask = _mm256_loadu_si256((const __m256i*)mask);
    return _mm256_maskload_pd(ptr, vmask);
}

inline void masked_storeu_pd(double* ptr, __m256d val, const int64_t* mask) {
    __m256i vmask = _mm256_loadu_si256((const __m256i*)mask);
    _mm256_maskstore_pd(ptr, vmask, val);
}

// Horizontal sum for __m256d (4 doubles)
inline double hsum256_pd(__m256d v) {
    // v = [d3, d2, d1, d0]
    __m256d v2 = _mm256_permute2f128_pd(v, v, 1); // [d1, d0, d3, d2] (swap 128-bit lanes)
    v = _mm256_add_pd(v, v2);                     // [d3+d1, d2+d0, d1+d3, d0+d2]
    __m256d v3 = _mm256_permute_pd(v, 0x5);       // shuffle within 128 lanes to swap pairs: 0b0101 -> swap
    v = _mm256_add_pd(v, v3);
    return _mm256_cvtsd_f64(v);
}

} 
