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

}
//                     Broadcasting & Dispatch Logic

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