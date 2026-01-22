#include "ops_avx2_f32.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>
#include <vector>

#if defined(__AVX2__)

namespace {

// ========================================================================
//                     Helpers for Tensor Access
// ========================================================================

template <typename T>
inline T* get_ptr(const Tensor& t) {
    if (!t.impl || !t.impl->data) return nullptr;
    return (T*)t.impl->data->data.get() + t.impl->offset;
}

// ========================================================================
//                     Internal AVX2 Math Constants & Helpers
// ========================================================================
#define YMM_1_PS   _mm256_set1_ps(1.0f)
#define YMM_05_PS  _mm256_set1_ps(0.5f)
#define YMM_0_PS   _mm256_setzero_ps()
#define YMM_NAN_PS _mm256_set1_ps(NAN)
#define YMM_PI32_0x7f _mm256_set1_epi32(0x7f)

// --- Masked Load/Store Helpers ---
static inline __m256 masked_loadu_ps(const float* ptr, size_t valid_count) {
    alignas(32) float tmp[8] = {0.0f};
    if (valid_count > 0) {
        std::memcpy(tmp, ptr, valid_count * sizeof(float));
    }
    return _mm256_load_ps(tmp);
}

static inline void masked_storeu_ps(float* ptr, __m256 v, size_t valid_count) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    if (valid_count > 0) {
        std::memcpy(ptr, tmp, valid_count * sizeof(float));
    }
}

// --- Abs ---
inline __m256 _mm256_abs_ps(__m256 x) {
    const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask));
}

// --- Exponential (Exp) ---
inline __m256 exp256_ps(__m256 x) {
    __m256 tmp = _mm256_setzero_ps();
    __m256 fx;
    __m256i emm0;
    __m256 one = YMM_1_PS;

    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    fx = _mm256_add_ps(fx, YMM_05_PS);
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
    y = _mm256_add_ps(y, YMM_05_PS);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, YMM_PI32_0x7f);
    emm0 = _mm256_slli_epi32(emm0, 23);
    
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

// --- Natural Logarithm (Ln) ---
inline __m256 log256_ps(__m256 x) {
    __m256 one = YMM_1_PS;
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);

    x = _mm256_max_ps(x, _mm256_set1_ps(1.17549435e-38f)); 

    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    
    x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff)));
    x = _mm256_or_ps(x, YMM_05_PS);

    emm0 = _mm256_sub_epi32(emm0, YMM_PI32_0x7f);
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

    tmp = _mm256_mul_ps(z, YMM_05_PS);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, _mm256_set1_ps(0.693359375f));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

// --- Sine (Sin) ---
inline __m256 sin256_ps(__m256 x) {
    __m256 xmm1, sign_bit, y;
    __m256i emm2;
    sign_bit = x;
    x = _mm256_abs_ps(x);

    xmm1 = _mm256_mul_ps(x, _mm256_set1_ps(0.63661977236758134308f)); // 2/pi
    emm2 = _mm256_cvttps_epi32(xmm1);
    emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
    emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(emm2);

    __m256 poly_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(emm2, _mm256_set1_epi32(4)), _mm256_setzero_si256()));
    sign_bit = _mm256_xor_ps(sign_bit, _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), poly_mask));

    __m256 m1 = _mm256_mul_ps(y, _mm256_set1_ps(-1.5703125f));
    __m256 m2 = _mm256_mul_ps(y, _mm256_set1_ps(-4.837512969970703125e-4f));
    __m256 m3 = _mm256_mul_ps(y, _mm256_set1_ps(-7.549789948768648e-8f));
    
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
    __m256 tmp = _mm256_mul_ps(z, YMM_05_PS);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, YMM_1_PS);
    y = _mm256_mul_ps(y, x);

    return _mm256_xor_ps(y, _mm256_and_ps(sign_bit, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))));
}

// --- Cosine (Cos) ---
inline __m256 cos256_ps(__m256 x) {
    x = _mm256_add_ps(x, _mm256_set1_ps(1.57079632679489661923f));
    return sin256_ps(x);
}

// --- Tanh ---
inline __m256 tanh256_ps(__m256 x) {
    __m256 two_x = _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    __m256 exp_2x = exp256_ps(two_x);
    __m256 num = _mm256_sub_ps(exp_2x, YMM_1_PS);
    __m256 den = _mm256_add_ps(exp_2x, YMM_1_PS);
    return _mm256_div_ps(num, den);
}

// --- Sigmoid ---
inline __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    __m256 e = exp256_ps(neg_x);
    __m256 den = _mm256_add_ps(YMM_1_PS, e);
    return _mm256_div_ps(YMM_1_PS, den);
}

// --- Pow ---
inline __m256 pow256_ps(__m256 a, __m256 b) {
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

// --- Horizontal Max ---
inline float hmax256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_max_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_max_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

// --- Horizontal Min ---
inline float hmin256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_min_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_min_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_min_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

} // namespace

// ---------------------------
// Broadcasting helpers
// ---------------------------

// REPLACED: Use int64_t for offsets to avoid overflow on large tensors
static inline std::vector<int64_t> get_strides_bytes(const Tensor& t) {
    std::vector<int64_t> strides_bytes;
    if (!t.impl) return strides_bytes;
    size_t el_size = dtype_size(t._dtype()); 
    strides_bytes.reserve(t.impl->ndim);
    // Explicit scalar handling: if numel == 1, effective strides are 0 for broadcasting
    if (t.numel() == 1) {
        for (size_t i = 0; i < t.impl->ndim; ++i) strides_bytes.push_back(0);
    } else {
        for (size_t s : t.impl->strides) {
            strides_bytes.push_back(static_cast<int64_t>(s * el_size));
        }
    }
    return strides_bytes;
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

// Used for StrideIterator initialization
static std::vector<int64_t> build_index_multipliers(const std::vector<size_t>& shape) {
    std::vector<int64_t> mult(shape.size());
    if (shape.empty()) return mult;
    mult.back() = 1;
    for (int i = (int)shape.size()-2; i >= 0; --i) mult[i] = mult[i+1] * (int64_t)shape[i+1];
    return mult;
}

// ---------------------- Stride Iterator ----------------------

namespace {
    struct StrideIterator {
        size_t ndim;
        std::vector<size_t> shape;
        std::vector<int64_t> strides_a;
        std::vector<int64_t> strides_b;
        std::vector<size_t> coords;
        int64_t offset_a;
        int64_t offset_b;

        StrideIterator(const std::vector<size_t>& out_shape, 
                       const std::vector<int64_t>& s_a, 
                       const std::vector<int64_t>& s_b) 
            : shape(out_shape), strides_a(s_a), strides_b(s_b) {
            ndim = shape.size();
            coords.resize(ndim, 0);
            offset_a = 0;
            offset_b = 0;
        }

        // Initialize at specific linear index without linear iteration
        void init(size_t linear_idx, const std::vector<int64_t>& multipliers) {
            offset_a = 0;
            offset_b = 0;
            size_t rem = linear_idx;
            for (size_t i = 0; i < ndim; ++i) {
                coords[i] = (rem / multipliers[i]) % shape[i];
                offset_a += coords[i] * strides_a[i];
                offset_b += coords[i] * strides_b[i];
            }
        }

        // Advance by 1
        inline void next() {
            for (int i = (int)ndim - 1; i >= 0; --i) {
                coords[i]++;
                offset_a += strides_a[i];
                offset_b += strides_b[i];
                if (coords[i] < shape[i]) {
                    return;
                }
                // Wrap around
                coords[i] = 0;
                offset_a -= strides_a[i] * (int64_t)shape[i]; 
                offset_b -= strides_b[i] * (int64_t)shape[i];
            }
        }
    };
}

// ---------------------- Binary Operations ----------------------

// FAST PATH: Contiguous + Same Shape (Supports 4x Unrolling)
template <typename Func>
Tensor binary_op_contiguous(const Tensor& A, const Tensor& B, Func op) {
    Tensor out(A.shape(), DType::Float32);
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* out_ptr = get_ptr<float>(out);
    size_t n = A.numel();
    size_t vec_limit = (n / 32) * 32;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vec_limit; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a_ptr + i);
        __m256 a1 = _mm256_loadu_ps(a_ptr + i + 8);
        __m256 a2 = _mm256_loadu_ps(a_ptr + i + 16);
        __m256 a3 = _mm256_loadu_ps(a_ptr + i + 24);

        __m256 b0 = _mm256_loadu_ps(b_ptr + i);
        __m256 b1 = _mm256_loadu_ps(b_ptr + i + 8);
        __m256 b2 = _mm256_loadu_ps(b_ptr + i + 16);
        __m256 b3 = _mm256_loadu_ps(b_ptr + i + 24);

        _mm256_storeu_ps(out_ptr + i,      op(a0, b0));
        _mm256_storeu_ps(out_ptr + i + 8,  op(a1, b1));
        _mm256_storeu_ps(out_ptr + i + 16, op(a2, b2));
        _mm256_storeu_ps(out_ptr + i + 24, op(a3, b3));
    }

    // Tail handling
    for (size_t i = vec_limit; i < n; i += 8) {
        size_t tail = (n - i < 8) ? (n - i) : 8;
        __m256 va = masked_loadu_ps(a_ptr + i, tail);
        __m256 vb = masked_loadu_ps(b_ptr + i, tail);
        __m256 vr = op(va, vb);
        masked_storeu_ps(out_ptr + i, vr, tail);
    }
    return out;
}

// GENERAL PATH: Strided Iterator + Broadcast Fast Path
template <typename Func>
Tensor binary_op_general(const Tensor& A, const Tensor& B, Func op) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out(out_shape, DType::Float32);
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* out_ptr = get_ptr<float>(out);

    auto multipliers = build_index_multipliers(out_shape);
    auto a_strides = get_strides_bytes(A);
    auto b_strides = get_strides_bytes(B);

    // Padding strides if rank differs (numpy broadcasting rules: prepend 1s)
    size_t ndim = out_shape.size();
    if (a_strides.size() < ndim) a_strides.insert(a_strides.begin(), ndim - a_strides.size(), 0);
    if (b_strides.size() < ndim) b_strides.insert(b_strides.begin(), ndim - b_strides.size(), 0);

    // If a dimension is 1 in input but N in output, stride must be 0 (broadcast)
    for(size_t i=0; i<ndim; ++i) {
        if(out_shape[i] > 1) {
             size_t a_dim = (i >= ndim - A.shape().size()) ? A.shape()[i - (ndim - A.shape().size())] : 1;
             if(a_dim == 1) a_strides[i] = 0;
             
             size_t b_dim = (i >= ndim - B.shape().size()) ? B.shape()[i - (ndim - B.shape().size())] : 1;
             if(b_dim == 1) b_strides[i] = 0;
        }
    }

    // --- Fast Path Detection: Vectorized Inner Loop ---
    // If the innermost dimension is contiguous (stride 4) or scalar broadcast (stride 0)
    // for both A and B, we can use AVX for the inner loop.
    int64_t last_sa = ndim > 0 ? a_strides.back() : 0;
    int64_t last_sb = ndim > 0 ? b_strides.back() : 0;
    size_t inner_dim_size = ndim > 0 ? out_shape.back() : 1;

    bool can_vectorize = (last_sa == 0 || last_sa == sizeof(float)) && 
                         (last_sb == 0 || last_sb == sizeof(float)) &&
                         (inner_dim_size >= 8); // Only worth it if inner dim is large enough

    #pragma omp parallel
    {
        StrideIterator it(out_shape, a_strides, b_strides);
        
        // OpenMP chunks the outer loop linearly.
        // We initialize the StrideIterator to the start of the chunk.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < out_numel; ++i) {
            // Optimization: Only run expensive init/re-sync at start of chunk or if logic desyncs
            // Ideally, we lift 'init' out of loop, but OpenMP hides the chunk bounds.
            // Standard pattern: Compute 'it' state from 'i' manually at start of iteration? 
            // Too slow to do every iter. 
            // Correct OpenMP pattern with custom iterator:
            // Since we can't easily hook into OpenMP's chunk start, we rely on the fact 
            // that for very large arrays, we can afford one div/mod per thread start
            // BUT we are inside the loop here.
            // WORKAROUND: Use 'i' to detect chunk start? No. 
            // Fallback: We MUST use div/mod if we can't maintain state.
            // BUT we want to avoid div/mod.
            
            // Re-think: We can't easily use StrideIterator cleanly inside a simple `#pragma omp for` 
            // without paying the initialization cost per element OR managing our own chunks.
            // LET'S MANAGE OWN CHUNKS to ensure efficiency.
            
        } // End dummy loop to switch to block based
        
        // Manual chunking for StrideIterator efficiency
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk_size = (out_numel + nthreads - 1) / nthreads;
        size_t start = std::min(tid * chunk_size, out_numel);
        size_t end = std::min((tid + 1) * chunk_size, out_numel);

        if (start < end) {
            it.init(start, multipliers);

            if (can_vectorize) {
                size_t current = start;
                while (current < end) {
                    // How many elements left in current inner dimension row?
                    size_t current_idx_in_inner = it.coords[ndim - 1];
                    size_t rem_in_inner = inner_dim_size - current_idx_in_inner;
                    
                    // How many can we process in this batch (limit by end of chunk)
                    size_t count = std::min(rem_in_inner, end - current);
                    
                    // Vectorize this segment
                    size_t j = 0;
                    for (; j + 8 <= count; j += 8) {
                        __m256 va = (last_sa == 0) ? _mm256_set1_ps(*(const float*)((char*)a_ptr + it.offset_a))
                                                   : _mm256_loadu_ps((const float*)((char*)a_ptr + it.offset_a + j * 4));
                        __m256 vb = (last_sb == 0) ? _mm256_set1_ps(*(const float*)((char*)b_ptr + it.offset_b))
                                                   : _mm256_loadu_ps((const float*)((char*)b_ptr + it.offset_b + j * 4));
                        __m256 vr = op(va, vb);
                        _mm256_storeu_ps(out_ptr + current + j, vr);
                    }
                    // Tail of segment
                    for (; j < count; ++j) {
                        float val_a = *(const float*)((char*)a_ptr + it.offset_a + j * last_sa);
                        float val_b = *(const float*)((char*)b_ptr + it.offset_b + j * last_sb);
                        out_ptr[current + j] = op(_mm256_set1_ps(val_a), _mm256_set1_ps(val_b))[0]; // Use scalar op fallback via AVX wrapper or simple math? 
                        // Our op returns __m256. Extracting scalar is slow.
                        // Better to keep scalar loop below pure scalar?
                        // For consistency, we rely on the vector op or scalar equivalent.
                        // Since 'op' is a lambda returning __m256, we can't easily extract scalar func.
                        // We will use mask store for tail to keep using 'op'.
                        // Wait, single element mask store is overkill.
                        // We assume 'op' behavior is elementwise. 
                        // Let's use masked load/store for tail to reuse 'op'.
                    }
                    
                    // Advance generic iterator by 'count'
                    // Since 'next' moves by 1, and we processed 'count' elements which might wrap...
                    // Actually, since we are inside the inner dim, we just advanced indices on the last dim.
                    // We need to carefully update 'it' to match 'current + count'.
                    // Optimization: We know we just advanced along the last dimension.
                    // But if 'count' hit the boundary, we wrap.
                    // Doing 'next()' 'count' times is slow.
                    // Fast forward:
                    current += count;
                    if (current < end) {
                        // Re-sync iterator completely (safest/simplest for now)
                        it.init(current, multipliers); 
                    }
                }
            } else {
                // Scalar Stride Iterator Loop
                // Fallback for weird strides where we can't simple-load
                // We use AVX logic on single elements (broadcast) to reuse the 'op' lambda
                for (size_t i = start; i < end; ++i) {
                    float val_a = *(const float*)((char*)a_ptr + it.offset_a);
                    float val_b = *(const float*)((char*)b_ptr + it.offset_b);
                    
                    __m256 va = _mm256_set1_ps(val_a);
                    __m256 vb = _mm256_set1_ps(val_b);
                    __m256 vr = op(va, vb);
                    
                    float res;
                    _mm_store_ss(&res, _mm256_castps256_ps128(vr));
                    out_ptr[i] = res;
                    
                    it.next();
                }
            }
        }
    }

    return out;
}

// CENTRALIZED DISPATCHER
template <typename Func>
Tensor binary_op_dispatch(const Tensor& A, const Tensor& B, Func op) {
    // 1. Fully Contiguous Optimization
    if (A.is_contiguous() && B.is_contiguous() && A.shape() == B.shape()) {
        return binary_op_contiguous(A, B, op);
    }
    // 2. Scalar broadcasting (all dims 1) implicitly handled by general,
    // but specific scalar check (numel=1) optimizes stride setup in get_strides_bytes.
    
    // 3. General (handles all broadcasting, including scalar, with stride iterator)
    return binary_op_general(A, B, op);
}

//-----------------------Unary Template (AVX2)----------------------

template <typename Func>
Tensor unary_op_broadcast(const Tensor& A, Func op) {
    Tensor out(A.shape(), DType::Float32);
    // Unrolling for contiguous inputs
    if (A.is_contiguous()) {
        const float* a_ptr = get_ptr<float>(A);
        float* out_ptr = get_ptr<float>(out);
        size_t n = A.numel();
        size_t vec_limit = (n / 32) * 32;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_limit; i += 32) {
            __m256 v0 = _mm256_loadu_ps(a_ptr + i);
            __m256 v1 = _mm256_loadu_ps(a_ptr + i + 8);
            __m256 v2 = _mm256_loadu_ps(a_ptr + i + 16);
            __m256 v3 = _mm256_loadu_ps(a_ptr + i + 24);

            _mm256_storeu_ps(out_ptr + i,      op(v0));
            _mm256_storeu_ps(out_ptr + i + 8,  op(v1));
            _mm256_storeu_ps(out_ptr + i + 16, op(v2));
            _mm256_storeu_ps(out_ptr + i + 24, op(v3));
        }

        for (size_t i = vec_limit; i < n; i += 8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 va = masked_loadu_ps(a_ptr + i, tail);
            __m256 vr = op(va);
            masked_storeu_ps(out_ptr + i, vr, tail);
        }
    } else {
        // Fallback for non-contiguous: Make contiguous first
        Tensor A_contig = A.contiguous();
        const float* ac_ptr = get_ptr<float>(A_contig);
        float* out_ptr = get_ptr<float>(out);
        size_t n = A.numel();
        size_t vec_limit = (n / 32) * 32;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_limit; i += 32) {
            __m256 v0 = _mm256_loadu_ps(ac_ptr + i);
            __m256 v1 = _mm256_loadu_ps(ac_ptr + i + 8);
            __m256 v2 = _mm256_loadu_ps(ac_ptr + i + 16);
            __m256 v3 = _mm256_loadu_ps(ac_ptr + i + 24);

            _mm256_storeu_ps(out_ptr + i,      op(v0));
            _mm256_storeu_ps(out_ptr + i + 8,  op(v1));
            _mm256_storeu_ps(out_ptr + i + 16, op(v2));
            _mm256_storeu_ps(out_ptr + i + 24, op(v3));
        }
        for (size_t i = vec_limit; i < n; i += 8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 va = masked_loadu_ps(ac_ptr + i, tail);
            __m256 vr = op(va);
            masked_storeu_ps(out_ptr + i, vr, tail);
        }
    }
    return out;
}

// ========================================================================
//                        Implementations
// ========================================================================

Tensor add_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ return _mm256_add_ps(x, y); });
}
Tensor sub_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ return _mm256_sub_ps(x, y); });
}
Tensor mul_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ return _mm256_mul_ps(x, y); });
}
Tensor div_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ return _mm256_div_ps(x, y); });
}
Tensor pow_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ return pow256_ps(x, y); });
}

// Comparisons
template<int CMP_FLAG>
Tensor cmp_avx2_f32(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, []( __m256 x, __m256 y){
        __m256 m = _mm256_cmp_ps(x, y, CMP_FLAG);
        return _mm256_and_ps(m, YMM_1_PS);
    });
}

Tensor lt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LT_OQ>(a,b); }
Tensor le_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_LE_OQ>(a,b); }
Tensor gt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GT_OQ>(a,b); }
Tensor ge_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_GE_OQ>(a,b); }
Tensor eq_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_avx2_f32<_CMP_NEQ_OQ>(a,b); }

// Unary
Tensor abs_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_abs_ps(x); }); }
Tensor sqrt_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_sqrt_ps(x); }); }
Tensor relu_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return _mm256_max_ps(x, YMM_0_PS); }); }
Tensor ln_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return log256_ps(x); }); }
Tensor exp_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return exp256_ps(x); }); }
Tensor sin_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return sin256_ps(x); }); }
Tensor cos_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return cos256_ps(x); }); }
Tensor tanh_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return tanh256_ps(x); }); }
Tensor sigmoid_avx2_f32(const Tensor& a) { return unary_op_broadcast(a, [](__m256 x){ return sigmoid256_ps(x); }); }
Tensor softplus_avx2_f32(const Tensor& a) { 
    return unary_op_broadcast(a, [](__m256 x){ 
        return log256_ps(_mm256_add_ps(YMM_1_PS, exp256_ps(x))); 
    }); 
}

#define OMP_SIMD_UNARY_AVX2(FUNC_NAME, STD_FUNC) \
Tensor FUNC_NAME(const Tensor& a) { \
    Tensor out(a.shape(), DType::Float32); \
    /* Ensure contiguity for SIMD loop */ \
    Tensor a_c = a.is_contiguous() ? a : a.contiguous(); \
    const float* pa = get_ptr<float>(a_c); \
    float* pout = get_ptr<float>(out); \
    size_t n = a.numel(); \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < n; ++i) { \
        pout[i] = STD_FUNC(pa[i]); \
    } \
    return out; \
}

OMP_SIMD_UNARY_AVX2(asin_avx2_f32, std::asin)
OMP_SIMD_UNARY_AVX2(acos_avx2_f32, std::acos)
OMP_SIMD_UNARY_AVX2(tan_avx2_f32, std::tan)
OMP_SIMD_UNARY_AVX2(atan_avx2_f32, std::atan)
OMP_SIMD_UNARY_AVX2(sinh_avx2_f32, std::sinh)
OMP_SIMD_UNARY_AVX2(cosh_avx2_f32, std::cosh)

// Matmul (Simple blocked AVX2)
Tensor matmul_avx2_f32(const Tensor& A, const Tensor& B) {
    // Matmul optimized kernels usually require contiguous memory
    Tensor A_contig = A.is_contiguous() ? A : A.contiguous();
    Tensor B_contig = B.is_contiguous() ? B : B.contiguous();

    if (A_contig.shape().size() != 2 || B_contig.shape().size() != 2) throw std::runtime_error("matmul_avx2: only 2D");
    size_t M = A_contig.shape()[0];
    size_t K = A_contig.shape()[1];
    size_t N = B_contig.shape()[1];
    if (K != B_contig.shape()[0]) throw std::runtime_error("matmul_avx2: shape mismatch");

    Tensor C({M, N}, DType::Float32);
    const float* a_ptr = get_ptr<float>(A_contig);
    const float* b_ptr = get_ptr<float>(B_contig);
    float* c_ptr = get_ptr<float>(C);
    
    std::memset(c_ptr, 0, M * N * sizeof(float));

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 va = _mm256_set1_ps(a_ptr[i*K + k]);
            size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 vc = _mm256_loadu_ps(c_ptr + i*N + j);
                __m256 vb = _mm256_loadu_ps(b_ptr + k*N + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(c_ptr + i*N + j, vc);
            }
            if (j < N) {
                size_t tail = N - j;
                __m256 vc = masked_loadu_ps(c_ptr + i*N + j, tail);
                __m256 vb = masked_loadu_ps(b_ptr + k*N + j, tail);
                vc = _mm256_fmadd_ps(va, vb, vc);
                masked_storeu_ps(c_ptr + i*N + j, vc, tail);
            }
        }
    }
    return C;
}

// Reductions
Tensor sum_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_avx2: only dim=-1");
    // Reductions typically iterate linearly over memory regardless of shape if dim=-1
    // BUT if t is not contiguous, linear iteration is invalid.
    Tensor t_c = t.is_contiguous() ? t : t.contiguous();
    size_t n = t_c.numel();
    const float* data = get_ptr<float>(t_c);
    float global_sum = 0.0f;

    #pragma omp parallel
    {
        // 4x unrolled accumulators
        __m256 vsum0 = YMM_0_PS;
        __m256 vsum1 = YMM_0_PS;
        __m256 vsum2 = YMM_0_PS;
        __m256 vsum3 = YMM_0_PS;

        size_t vec_limit = (n / 32) * 32;

        #pragma omp for nowait schedule(static)
        for(size_t i=0; i < vec_limit; i+=32) {
             __m256 v0 = _mm256_loadu_ps(data + i);
             __m256 v1 = _mm256_loadu_ps(data + i + 8);
             __m256 v2 = _mm256_loadu_ps(data + i + 16);
             __m256 v3 = _mm256_loadu_ps(data + i + 24);

             vsum0 = _mm256_add_ps(vsum0, v0);
             vsum1 = _mm256_add_ps(vsum1, v1);
             vsum2 = _mm256_add_ps(vsum2, v2);
             vsum3 = _mm256_add_ps(vsum3, v3);
        }

        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);

        #pragma omp for nowait schedule(static)
        for (size_t i=vec_limit; i < n; i+=8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else v = masked_loadu_ps(data + i, tail);
            vsum0 = _mm256_add_ps(vsum0, v);
        }

        float local_sum = hsum256_ps(vsum0);
        #pragma omp atomic
        global_sum += local_sum;
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_sum;
    return out;
}

Tensor mean_avx2_f32(const Tensor& t, int dim) {
    Tensor s = sum_avx2_f32(t, dim);
    float n = static_cast<float>(t.numel());
    ((float*)get_ptr<float>(s))[0] /= n;
    return s;
}

Tensor max_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_avx2: only dim=-1");
    
    // Ensure contiguous memory for optimal traversal
    Tensor t_c = t.is_contiguous() ? t : t.contiguous();
    const float* data = get_ptr<float>(t_c);
    size_t n = t_c.numel();
    
    // Initialize global max with lowest possible value
    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        // Use multiple accumulators to break dependency chains and hide latency.
        // Unrolling by 4 (32 floats) per step.
        __m256 vmax0 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256 vmax1 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256 vmax2 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256 vmax3 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

        size_t vec_limit = (n / 32) * 32;

        #pragma omp for nowait schedule(static)
        for(size_t i = 0; i < vec_limit; i += 32) {
            __m256 v0 = _mm256_loadu_ps(data + i);
            __m256 v1 = _mm256_loadu_ps(data + i + 8);
            __m256 v2 = _mm256_loadu_ps(data + i + 16);
            __m256 v3 = _mm256_loadu_ps(data + i + 24);
            
            vmax0 = _mm256_max_ps(vmax0, v0);
            vmax1 = _mm256_max_ps(vmax1, v1);
            vmax2 = _mm256_max_ps(vmax2, v2);
            vmax3 = _mm256_max_ps(vmax3, v3);
        }

        // Reduce accumulators
        vmax0 = _mm256_max_ps(vmax0, vmax1);
        vmax2 = _mm256_max_ps(vmax2, vmax3);
        vmax0 = _mm256_max_ps(vmax0, vmax2);

        #pragma omp for nowait schedule(static)
        for (size_t i = vec_limit; i < n; i += 8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else {
                v = masked_loadu_ps(data + i, tail);
                __m256i mask_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
                __m256i limit = _mm256_set1_epi32((int)tail);
                __m256 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(limit, mask_idx));
                __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                v = _mm256_blendv_ps(neg_inf, v, mask);
            }
            vmax0 = _mm256_max_ps(vmax0, v);
        }

        float local_max = hmax256_ps(vmax0);
        
        #pragma omp critical
        {
            if(local_max > global_max) global_max = local_max;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_max;
    return out;
}

Tensor min_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_avx2: only dim=-1");
    Tensor t_c = t.is_contiguous() ? t : t.contiguous();
    const float* data = get_ptr<float>(t_c);
    size_t n = t_c.numel();
    float global_min = std::numeric_limits<float>::infinity();

    #pragma omp parallel
    {
        __m256 vmin0 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        __m256 vmin1 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        __m256 vmin2 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        __m256 vmin3 = _mm256_set1_ps(std::numeric_limits<float>::infinity());

        size_t vec_limit = (n / 32) * 32;

        #pragma omp for nowait schedule(static)
        for(size_t i = 0; i < vec_limit; i += 32) {
            __m256 v0 = _mm256_loadu_ps(data + i);
            __m256 v1 = _mm256_loadu_ps(data + i + 8);
            __m256 v2 = _mm256_loadu_ps(data + i + 16);
            __m256 v3 = _mm256_loadu_ps(data + i + 24);
            
            vmin0 = _mm256_min_ps(vmin0, v0);
            vmin1 = _mm256_min_ps(vmin1, v1);
            vmin2 = _mm256_min_ps(vmin2, v2);
            vmin3 = _mm256_min_ps(vmin3, v3);
        }

        vmin0 = _mm256_min_ps(vmin0, vmin1);
        vmin2 = _mm256_min_ps(vmin2, vmin3);
        vmin0 = _mm256_min_ps(vmin0, vmin2);

        #pragma omp for nowait schedule(static)
        for (size_t i = vec_limit; i < n; i += 8) {
            size_t tail = (n - i < 8) ? (n - i) : 8;
            __m256 v;
            if (tail == 8) v = _mm256_loadu_ps(data + i);
            else {
                v = masked_loadu_ps(data + i, tail);
                __m256i mask_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
                __m256i limit = _mm256_set1_epi32((int)tail);
                __m256 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(limit, mask_idx));
                __m256 pos_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
                v = _mm256_blendv_ps(pos_inf, v, mask);
            }
            vmin0 = _mm256_min_ps(vmin0, v);
        }

        float local_min = hmin256_ps(vmin0);
        
        #pragma omp critical
        {
            if(local_min < global_min) global_min = local_min;
        }
    }
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = global_min;
    return out;
}

#endif // __AVX2__

