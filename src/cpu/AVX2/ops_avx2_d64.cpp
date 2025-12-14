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

/*----------------------Broadcasting & Dispatch Helpers---------------------------*/

static inline std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = sizeof(double);
    for (int i = (int)shape.size()-2; i >= 0; --i) {
        strides[i] = strides[i+1] * (int64_t)shape[i+1];
    }
    return strides;
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t ai = (i < n - na) ? 1 : a[i - (n - na)];
        size_t bi = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (ai != 1 && bi != 1 && ai != bi) throw std::runtime_error("broadcast: incompatible shapes");
        out[i] = std::max(ai, bi);
    }
    return out;
}

static std::vector<int64_t> build_index_multipliers(const std::vector<size_t>& shape) {
    std::vector<int64_t> mult(shape.size());
    if (shape.empty()) return mult;
    mult.back() = 1;
    for (int i = (int)shape.size()-2; i >= 0; --i) mult[i] = mult[i+1] * (int64_t)shape[i+1];
    return mult;
}

// Compute byte offset for double (stride is 8 bytes)
static inline int32_t compute_offset_bytes(size_t lin_idx, const std::vector<size_t>& out_shape, const std::vector<int64_t>& out_mult, const std::vector<size_t>& in_shape, const std::vector<int64_t>& in_strides_bytes) {
    int32_t offset = 0;
    size_t nd = out_shape.size();
    size_t offset_dim = nd - in_shape.size();
    for (size_t d = 0; d < nd; ++d) {
        size_t coord = (lin_idx / out_mult[d]) % out_shape[d];
        size_t in_coord = 0;
        if (d >= offset_dim) {
            size_t idx = d - offset_dim;
            if (in_shape[idx] != 1) in_coord = coord;
            offset += (int32_t)(in_coord * in_strides_bytes[idx]);
        }
    }
    return offset;
}
