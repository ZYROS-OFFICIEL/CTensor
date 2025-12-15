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


// --- Sine (Sin) for AVX-512 ---
inline __m512 sin512_ps(__m512 x) {
    __m512 xmm1, sign_bit, y;
    __m512i emm2;
    sign_bit = x;
    x = _mm512_abs_ps(x);

    xmm1 = _mm512_mul_ps(x, _mm512_set1_ps(0.63661977236758134308f)); // 2/pi
    emm2 = _mm512_cvttps_epi32(xmm1);
    emm2 = _mm512_add_epi32(emm2, _mm512_set1_epi32(1));
    emm2 = _mm512_and_si512(emm2, _mm512_set1_epi32(~1));
    y = _mm512_cvtepi32_ps(emm2);

    __mmask16 poly_mask = _mm512_cmpeq_epi32_mask(_mm512_and_si512(emm2, _mm512_set1_epi32(4)), _mm512_setzero_si512());
    
    // sign bit logic
    __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
    // If poly_mask is true, we keep original sign. If false, we flip. 
    // Wait, standard Cephes logic:
    // swap_sign_bit = (emm2 & 4) != 0
    // if swap, sign = ^sign
    __m512 swap_sign = _mm512_and_ps(sign_mask, _mm512_mask_blend_ps(poly_mask, _zmm_1, _zmm_0)); // Logic slightly complex to port 1:1 visually, simplifying:
    
    // Correct logic: if (emm2 & 4) == 0 (poly_mask=1), no swap.
    // If (emm2 & 4) != 0 (poly_mask=0), swap.
    // We want to XOR with 0x80000000 if poly_mask is 0.
    __m512 xor_mask = _mm512_mask_blend_ps(poly_mask, sign_mask, _zmm_0); 
    sign_bit = _mm512_xor_ps(sign_bit, xor_mask);

    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-1.5703125f), x);
    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-4.837512969970703125e-4f), x);
    x = _mm512_fmadd_ps(y, _mm512_set1_ps(-7.549789948768648e-8f), x);

    __m512 z = _mm512_mul_ps(x, x);
    y = _mm512_set1_ps(2.443315711809948E-005f);
    y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(-1.388731625493765E-003f));
    y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(4.166664568298827E-002f));
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, _zmm_05, y); // y - 0.5*z
    y = _mm512_add_ps(y, _zmm_1);
    y = _mm512_mul_ps(y, x);

    return _mm512_xor_ps(y, _mm512_and_ps(sign_bit, sign_mask));
}

inline __m512 cos512_ps(__m512 x) {
    x = _mm512_add_ps(x, _mm512_set1_ps(1.57079632679489661923f));
    return sin512_ps(x);
}

inline __m512 tanh512_ps(__m512 x) {
    __m512 two_x = _mm512_mul_ps(x, _mm512_set1_ps(2.0f));
    __m512 exp_2x = exp512_ps(two_x);
    __m512 num = _mm512_sub_ps(exp_2x, _zmm_1);
    __m512 den = _mm512_add_ps(exp_2x, _zmm_1);
    return _mm512_div_ps(num, den);
}

inline __m512 sigmoid512_ps(__m512 x) {
    __m512 neg_x = _mm512_xor_ps(x, _mm512_set1_ps(-0.0f));
    __m512 e = exp512_ps(neg_x);
    __m512 den = _mm512_add_ps(_zmm_1, e);
    return _mm512_div_ps(_zmm_1, den);
}

inline __m512 pow512_ps(__m512 a, __m512 b) {
    return exp512_ps(_mm512_mul_ps(b, log512_ps(a)));
}

// Horizontal Sum for ZMM
inline float hsum512_ps(__m512 v) {
    // reduce to 256
    __m256 vlow = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    vlow = _mm256_add_ps(vlow, vhigh);
    // reduce to 128
    __m128 xlow = _mm256_castps256_ps128(vlow);
    __m128 xhigh = _mm256_extractf128_ps(vlow, 1);
    xlow = _mm_add_ps(xlow, xhigh);
    // reduce 128
    __m128 shuf = _mm_movehdup_ps(xlow);
    __m128 sums = _mm_add_ps(xlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

} // namespace

// ---------------------------Broadcasting & Dispatch Logic---------------------------

static inline std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = sizeof(float);
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