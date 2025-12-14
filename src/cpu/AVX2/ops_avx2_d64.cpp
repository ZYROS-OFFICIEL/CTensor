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
/*----------------------Generic Binary Template (Double)---------------------------*/

Tensor binary_op_broadcast_d64(const Tensor& A, const Tensor& B, std::function<__m256d(__m256d,__m256d)> avx_func) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out(out_shape, A.device(), DType::Double64);

    const double* a_ptr = (const double*)A.data();
    const double* b_ptr = (const double*)B.data();
    double* out_ptr = (double*)out.data();

    auto out_mult = build_index_multipliers(out_shape);
    auto a_strides = shape_to_strides_bytes(a_shape);
    auto b_strides = shape_to_strides_bytes(b_shape);

    bool a_contig = (A.is_contiguous()) && (a_shape == out_shape);
    bool b_contig = (B.is_contiguous()) && (b_shape == out_shape);
    bool a_is_scalar = (A.numel() == 1);
    bool b_is_scalar = (B.numel() == 1);

    // Vector width for double is 4
    size_t vec_end = (out_numel / 4) * 4;

    #pragma omp parallel
    {
        int32_t gather_idx[4]; // buffer for 4 offsets

        #pragma omp for
        for (size_t i = 0; i < vec_end; i += 4) {
            __m256d va, vb;

            // Load A
            if (a_contig) {
                va = _mm256_loadu_pd(a_ptr + i);
            } else if (a_is_scalar) {
                va = _mm256_set1_pd(a_ptr[0]);
            } else {
                for (int lane = 0; lane < 4; ++lane)
                    gather_idx[lane] = compute_offset_bytes(i + lane, out_shape, out_mult, a_shape, a_strides);
                // i32gather_pd gathers doubles using int32 indices * scale. 
                // Since our offsets are bytes, we can't use scale=8 unless indices were element indices.
                // But i32gather_pd takes indices as Bytes if scale=1. 
                // _mm256_i32gather_pd(double const * base_addr, __m128i vindex, const int scale)
                // vindex is 4 integers (XMM register).
                __m128i vidx = _mm_loadu_si128((const __m128i*)gather_idx);
                va = _mm256_i32gather_pd(a_ptr, vidx, 1);
            }

            // Load B
            if (b_contig) {
                vb = _mm256_loadu_pd(b_ptr + i);
            } else if (b_is_scalar) {
                vb = _mm256_set1_pd(b_ptr[0]);
            } else {
                for (int lane = 0; lane < 4; ++lane)
                    gather_idx[lane] = compute_offset_bytes(i + lane, out_shape, out_mult, b_shape, b_strides);
                __m128i vidx = _mm_loadu_si128((const __m128i*)gather_idx);
                vb = _mm256_i32gather_pd(b_ptr, vidx, 1);
            }

            // Op
            __m256d vr = avx_func(va, vb);
            _mm256_storeu_pd(out_ptr + i, vr);
        }

        // Tail
        size_t tail_start = vec_end;
        size_t tail_count = out_numel - tail_start;
        if (tail_count > 0) {
            int64_t mask[4];
            build_tail_mask_d64(mask, tail_count);

            __m256d va_tail, vb_tail;
            
            // Tail Load A
            if (a_contig) {
                va_tail = masked_loadu_pd(a_ptr + tail_start, mask);
            } else if (a_is_scalar) {
                va_tail = _mm256_set1_pd(a_ptr[0]);
            } else {
                double tmp[4] = {0};
                for(size_t j=0; j<tail_count; ++j) {
                    size_t offset = compute_offset_bytes(tail_start + j, out_shape, out_mult, a_shape, a_strides);
                    // offset is in bytes
                    tmp[j] = *(const double*)((const char*)a_ptr + offset);
                }
                va_tail = _mm256_loadu_pd(tmp);
            }

            // Tail Load B
            if (b_contig) {
                vb_tail = masked_loadu_pd(b_ptr + tail_start, mask);
            } else if (b_is_scalar) {
                vb_tail = _mm256_set1_pd(b_ptr[0]);
            } else {
                double tmp[4] = {0};
                for(size_t j=0; j<tail_count; ++j) {
                    size_t offset = compute_offset_bytes(tail_start + j, out_shape, out_mult, b_shape, b_strides);
                    tmp[j] = *(const double*)((const char*)b_ptr + offset);
                }
                vb_tail = _mm256_loadu_pd(tmp);
            }

            __m256d vr_tail = avx_func(va_tail, vb_tail);
            masked_storeu_pd(out_ptr + tail_start, vr_tail, mask);
        }
    }

    return out;
}
