#include "ops_avx2_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <functional>

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
// --- Horizontal Sum ---
inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

} 
// ---------------------------
// Broadcasting helpers
// ---------------------------

// Convert shape vector to strides in bytes
static inline std::vector<int64_t> shape_to_strides_bytes(const std::vector<size_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = sizeof(float);
    for (int i = (int)shape.size()-2; i >= 0; --i) {
        strides[i] = strides[i+1] * (int64_t)shape[i+1];
    }
    return strides;
}

// Compute broadcasted output shape for two shapes
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

// Build per-dimension index multipliers for linear index -> offset
static std::vector<int64_t> build_index_multipliers(const std::vector<size_t>& shape) {
    std::vector<int64_t> mult(shape.size());
    if (shape.empty()) return mult;
    mult.back() = 1;
    for (int i = (int)shape.size()-2; i >= 0; --i) mult[i] = mult[i+1] * (int64_t)shape[i+1];
    return mult;
}

// Given linear index, produce offset in bytes using shape/strides (strides in bytes)
static inline int32_t compute_offset_bytes(size_t lin_idx, const std::vector<size_t>& out_shape, const std::vector<int64_t>& out_mult, const std::vector<size_t>& in_shape, const std::vector<int64_t>& in_strides_bytes) {
    int32_t offset = 0;
    size_t nd = out_shape.size();
    size_t offset_dim = nd - in_shape.size();
    for (size_t d = 0; d < nd; ++d) {
        size_t coord = (lin_idx / out_mult[d]) % out_shape[d];
        // map to input coord
        size_t in_coord = 0;
        if (d < offset_dim) {
            in_coord = 0; // leading dims are broadcast
        } else {
            size_t idx = d - offset_dim;
            if (in_shape[idx] == 1) in_coord = 0;
            else in_coord = coord;
        }
        if (d >= offset_dim) {
            size_t idx = d - offset_dim;
            offset += (int32_t)(in_coord * (in_strides_bytes[idx]));
        }
    }
    return offset;
}

// ---------------------------
// Core templates using broadcasting and gathers/masked loads
// No scalar fallback (caller handles any special-cases)
// ---------------------------

// Binary op with broadcasting. avx_func takes (__m256 a, __m256 b) and returns __m256.
// scalar_func is provided only for correctness reference (not used here because caller has dispatcher) but kept for signature compat.
Tensor binary_op_broadcast(const Tensor& A, const Tensor& B, std::function<__m256(__m256,__m256)> avx_func) {
    // Extract shapes and compute broadcasted shape
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out(out_shape, DType::Float32);

    const float* a_ptr = (const float*)A.impl->data->data.get();
    const float* b_ptr = (const float*)B.impl->data->data.get();
    float* out_ptr = (float*)out.impl->data->data.get();

    // build index multipliers and strides (bytes)
    auto out_mult = build_index_multipliers(out_shape);
    auto a_strides = shape_to_strides_bytes(a_shape);
    auto b_strides = shape_to_strides_bytes(b_shape);
    auto a_nd = a_shape.size();
    auto b_nd = b_shape.size();
    size_t nd = out_shape.size();

    // precompute whether a/b are contiguous with element stride == sizeof(float) and not broadcasted
    bool a_contig = (A.is_contiguous());
    bool b_contig = (B.is_contiguous());

    // We'll iterate in blocks of 8 and use gathers for non-unit stride cases
    size_t vec_end = (out_numel / 8) * 8;
    int32_t tail_maskbits[8];
    build_tail_mask(tail_maskbits, out_numel - vec_end);

    // Parallel loop
    #pragma omp parallel
    {
        // thread-local gather index buffer
        int32_t gather_idx[8];

        #pragma omp for
        for (size_t i = 0; i < vec_end; i += 8) {
            // for each lane compute byte offsets for a and b
            for (int lane = 0; lane < 8; ++lane) {
                size_t lin = i + lane;
                gather_idx[lane] = compute_offset_bytes(lin, out_shape, out_mult, a_shape, a_strides);
            }
            // decide load method for A
            __m256 va;
            bool a_is_broadcast = (a_shape.size()==1 && a_shape[0]==1) || (a_shape.size()==0);
            if (a_contig && a_shape == out_shape) {
                va = _mm256_loadu_ps(a_ptr + i);
            } else if (a_is_broadcast) {
                // scalar broadcast
                float aval = *((const float*)a_ptr);
                va = _mm256_set1_ps(aval);
            } else {
                // gather
                // gather expects offsets in bytes; use i32 offsets
                va = _mm256_i32gather_ps((const float*)a_ptr, _mm256_loadu_si256((const __m256i*)gather_idx), 1);
            }

            // compute offsets for B
            for (int lane = 0; lane < 8; ++lane) {
                size_t lin = i + lane;
                gather_idx[lane] = compute_offset_bytes(lin, out_shape, out_mult, b_shape, b_strides);
            }
            __m256 vb;
            bool b_is_broadcast = (b_shape.size()==1 && b_shape[0]==1) || (b_shape.size()==0);
            if (b_contig && b_shape == out_shape) {
                vb = _mm256_loadu_ps(b_ptr + i);
            } else if (b_is_broadcast) {
                float bval = *((const float*)b_ptr);
                vb = _mm256_set1_ps(bval);
            } else {
                vb = _mm256_i32gather_ps((const float*)b_ptr, _mm256_loadu_si256((const __m256i*)gather_idx), 1);
            }

            __m256 vr = avx_func(va, vb);
            _mm256_storeu_ps(out_ptr + i, vr);
        }

        // handle tail if any (masked store)
        size_t tail_start = vec_end;
        size_t tail = out_numel - vec_end;
        if (tail) {
            // compute gather offsets for tail lanes
            int32_t tail_idx[8];
            for (size_t lane = 0; lane < tail; ++lane) tail_idx[lane] = compute_offset_bytes(vec_end + lane, out_shape, out_mult, a_shape, a_strides);
            // load a (use gather or broadcast)
            __m256 va;
            if (a_contig && a_shape == out_shape) {
                // load first tail elements into vector with masked load
                int32_t maskbits[8]; build_tail_mask(maskbits, tail);
                va = masked_loadu_ps(a_ptr + vec_end, maskbits);
            } else if ((a_shape.size()==1 && a_shape[0]==1) || (a_shape.size()==0)) {
                va = _mm256_set1_ps(*((const float*)a_ptr));
            } else {
                // build gather idx bytes for tail
                int32_t gidx[8]; for (size_t lane=0; lane<tail; ++lane) gidx[lane] = tail_idx[lane]; for (size_t lane=tail; lane<8; ++lane) gidx[lane]=0;
                va = _mm256_i32gather_ps((const float*)a_ptr, _mm256_loadu_si256((const __m256i*)gidx), 1);
            }
            // similarly for B
            if (b_contig && b_shape == out_shape) {
                int32_t maskbits[8]; build_tail_mask(maskbits, tail);
                __m256 vb = masked_loadu_ps(b_ptr + vec_end, maskbits);
                __m256 vr = avx_func(va, vb);
                int32_t maskbits_store[8]; build_tail_mask(maskbits_store, tail);
                masked_storeu_ps(out_ptr + vec_end, vr, maskbits_store);
            } else if ((b_shape.size()==1 && b_shape[0]==1) || (b_shape.size()==0)) {
                __m256 vb = _mm256_set1_ps(*((const float*)b_ptr));
                __m256 vr = avx_func(va, vb);
                int32_t maskbits_store[8]; build_tail_mask(maskbits_store, tail);
                masked_storeu_ps(out_ptr + vec_end, vr, maskbits_store);
            } else {
                int32_t gidxb[8]; for (size_t lane=0; lane<tail; ++lane) gidxb[lane] = compute_offset_bytes(vec_end + lane, out_shape, out_mult, b_shape, b_strides); for (size_t lane=tail; lane<8; ++lane) gidxb[lane]=0;
                __m256 vb = _mm256_i32gather_ps((const float*)b_ptr, _mm256_loadu_si256((const __m256i*)gidxb), 1);
                __m256 vr = avx_func(va, vb);
                int32_t maskbits_store[8]; build_tail_mask(maskbits_store, tail);
                masked_storeu_ps(out_ptr + vec_end, vr, maskbits_store);
            }
        }
    }

    return out;
}


// Simple unary op with broadcasting (mostly for broadcasting scalar -> tensor or same shape)
Tensor unary_op_broadcast(const Tensor& A, std::function<__m256(__m256)> avx_func) {
    std::vector<size_t> a_shape = A.shape();
    size_t n = A.numel();
    Tensor out(a_shape, DType::Float32);
    const float* a_ptr = (const float*)A.impl->data->data.get();
    float* out_ptr = (float*)out.impl->data->data.get();

    size_t vec_end = (n / 8) * 8;
    int32_t tail_maskbits[8]; build_tail_mask(tail_maskbits, n - vec_end);

    #pragma omp parallel for
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a_ptr + i);
        __m256 vr = avx_func(va);
        _mm256_storeu_ps(out_ptr + i, vr);
    }
    if (n != vec_end) {
        int tail = (int)(n - vec_end);
        int32_t maskbits[8]; build_tail_mask(maskbits, tail);
        __m256 va = masked_loadu_ps(a_ptr + vec_end, maskbits);
        __m256 vr = avx_func(va);
        masked_storeu_ps(out_ptr + vec_end, vr, maskbits);
    }
    return out;
}

// ---------------------------
// Public API wrappers (vectorized)
// ---------------------------

Tensor add_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_add_ps(x, y); });
}
Tensor sub_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_sub_ps(x, y); });
}
Tensor mul_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_mul_ps(x, y); });
}
Tensor div_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return _mm256_div_ps(x, y); });
}
Tensor pow_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, [](__m256 x, __m256 y){ return pow256_ps(x, y); });
}


// comparisons: convert mask -> 0.0f / 1.0f
template<int CMP_FLAG>
Tensor cmp_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast(a, b, []( __m256 x, __m256 y){
        __m256 m = _mm256_cmp_ps(x, y, CMP_FLAG);
        // mask bits are 0xFFFFFFFF for true; convert to 1.0f by ANDing with 1.0
        return _mm256_and_ps(m, _ps_1);
    });
}

Tensor lt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LT_OQ>(a,b); }
Tensor le_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LE_OQ>(a,b); }
Tensor gt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GT_OQ>(a,b); }
Tensor ge_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GE_OQ>(a,b); }
Tensor eq_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_NEQ_OQ>(a,b); }

#endif