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

template <typename T>
inline T* get_ptr(const Tensor& t) {
    if (!t.impl || !t.impl->data) return nullptr;
    return (T*)t.impl->data->data.get() + t.impl->offset;
}

static const size_t AVX_OMP_THRESHOLD = 32768; 

// ========================================================================
//                     Internal AVX2 Helpers
// ========================================================================
#define YMM_1_PS   _mm256_set1_ps(1.0f)
#define YMM_0_PS   _mm256_setzero_ps()

static inline __m256 masked_loadu_ps(const float* ptr, size_t valid_count) {
    if (valid_count == 0) return YMM_0_PS;
    __m256i mask_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    __m256i limit = _mm256_set1_epi32((int)valid_count);
    __m256i mask_i = _mm256_cmpgt_epi32(limit, mask_idx);
    return _mm256_maskload_ps(ptr, mask_i);
}

static inline void masked_storeu_ps(float* ptr, __m256 v, size_t valid_count) {
    if (valid_count == 0) return;
    __m256i mask_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    __m256i limit = _mm256_set1_epi32((int)valid_count);
    __m256i mask_i = _mm256_cmpgt_epi32(limit, mask_idx);
    _mm256_maskstore_ps(ptr, mask_i, v);
}

// ---------------------- Optimized Odometer Iterator ----------------------
// Replaces integer division with incremental additions
// Supports up to 32 dims on stack, throws if exceeded
static const int MAX_ITER_DIMS = 32;

struct OdometerIterator {
    int ndim;
    const size_t* shape;
    size_t strides_a[MAX_ITER_DIMS]; 
    size_t strides_b[MAX_ITER_DIMS];
    size_t coords[MAX_ITER_DIMS];
    size_t offset_a;
    size_t offset_b;

    // Reset offsets when a dimension wraps around
    size_t back_strides_a[MAX_ITER_DIMS]; 
    size_t back_strides_b[MAX_ITER_DIMS];

    OdometerIterator(int nd, const size_t* sh, const std::vector<int64_t>& sa, const std::vector<int64_t>& sb) 
        : ndim(nd), shape(sh), offset_a(0), offset_b(0) {
        
        if (nd > MAX_ITER_DIMS) throw std::runtime_error("Odometer: rank too high for stack buffer");

        for(int i=0; i<ndim; ++i) {
            strides_a[i] = sa[i];
            strides_b[i] = sb[i];
            coords[i] = 0;
            // Precalculate backsteps
            back_strides_a[i] = sa[i] * (sh[i] - 1);
            back_strides_b[i] = sb[i] * (sh[i] - 1);
        }
    }
    
    // Initialize to arbitrary linear index (expensive, call once per chunk)
    void init(size_t linear_idx) {
        offset_a = 0; 
        offset_b = 0;
        size_t rem = linear_idx;
        for (int i = ndim - 1; i >= 0; --i) {
            size_t sz = shape[i];
            size_t c = rem % sz;
            rem /= sz;
            coords[i] = c;
            offset_a += c * strides_a[i];
            offset_b += c * strides_b[i];
        }
    }

    // Fast increment
    inline void next() {
        for (int i = ndim - 1; i >= 0; --i) {
            coords[i]++;
            offset_a += strides_a[i];
            offset_b += strides_b[i];
            
            if (coords[i] < shape[i]) {
                return;
            }
            // Wrap
            coords[i] = 0;
            offset_a -= (back_strides_a[i] + strides_a[i]); 
            offset_b -= (back_strides_b[i] + strides_b[i]);
        }
    }
};

} // namespace

// ---------------------- Binary Operations ----------------------

static void get_broadcast_strides(const Tensor& t, const std::vector<size_t>& out_shape, std::vector<int64_t>& strides) {
    int out_ndim = (int)out_shape.size();
    strides.assign(out_ndim, 0);
    int t_ndim = t.impl->ndim;
    int offset = out_ndim - t_ndim;
    
    size_t el_size = sizeof(float);
    
    for (int i = 0; i < t_ndim; ++i) {
        if (t.impl->shape[i] > 1) {
            strides[i + offset] = t.impl->strides[i] * el_size;
        }
    }
}

template <typename Func>
Tensor binary_op_general(const Tensor& A, const Tensor& B, Func op) {
    // 1. Compute output shape
    int an = A.impl->ndim;
    int bn = B.impl->ndim;
    int ndim = std::max(an, bn);
    std::vector<size_t> out_shape(ndim);
    
    for (int i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - an) ? 1 : A.impl->shape[i - (ndim - an)];
        size_t db = (i < ndim - bn) ? 1 : B.impl->shape[i - (ndim - bn)];
        if (da != db && da != 1 && db != 1) throw std::runtime_error("Shape mismatch");
        out_shape[i] = std::max(da, db);
    }
    
    Tensor out(out_shape, DType::Float32);
    size_t numel = out.numel();
    
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* out_ptr = get_ptr<float>(out);

    std::vector<int64_t> str_a, str_b;
    get_broadcast_strides(A, out_shape, str_a);
    get_broadcast_strides(B, out_shape, str_b);
    
    if (numel < AVX_OMP_THRESHOLD) {
        OdometerIterator it(ndim, out_shape.data(), str_a, str_b);
        for (size_t i = 0; i < numel; ++i) {
            float val_a = *(const float*)((const char*)a_ptr + it.offset_a);
            float val_b = *(const float*)((const char*)b_ptr + it.offset_b);
            
            __m256 va = _mm256_set1_ps(val_a);
            __m256 vb = _mm256_set1_ps(val_b);
            __m256 vr = op(va, vb);
            float res;
            _mm_store_ss(&res, _mm256_castps256_ps128(vr));
            out_ptr[i] = res;
            it.next();
        }
    } else {
        #pragma omp parallel
        {
            OdometerIterator it(ndim, out_shape.data(), str_a, str_b);
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = (numel + nthreads - 1) / nthreads;
            size_t start = std::min(chunk * tid, numel);
            size_t end = std::min(chunk * (tid + 1), numel);
            
            if (start < end) {
                it.init(start); 
                for (size_t i = start; i < end; ++i) {
                    float val_a = *(const float*)((const char*)a_ptr + it.offset_a);
                    float val_b = *(const float*)((const char*)b_ptr + it.offset_b);
                    
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

// Optimized Contiguous Path
template <typename Func>
Tensor binary_op_contiguous(const Tensor& A, const Tensor& B, Func op) {
    Tensor out(A.shape(), DType::Float32);
    const float* a_ptr = get_ptr<float>(A);
    const float* b_ptr = get_ptr<float>(B);
    float* out_ptr = get_ptr<float>(out);
    size_t n = A.numel();

    if (n < AVX_OMP_THRESHOLD) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + i);
            __m256 vb = _mm256_loadu_ps(b_ptr + i);
            _mm256_storeu_ps(out_ptr + i, op(va, vb));
        }
        for (; i < n; ++i) {
             float va = a_ptr[i];
             float vb = b_ptr[i];
             __m256 vva = _mm256_set1_ps(va);
             __m256 vvb = _mm256_set1_ps(vb);
             __m256 vr = op(vva, vvb);
             float res;
             _mm_store_ss(&res, _mm256_castps256_ps128(vr));
             out_ptr[i] = res;
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i += 8) {
            if (i + 8 <= n) {
                __m256 va = _mm256_loadu_ps(a_ptr + i);
                __m256 vb = _mm256_loadu_ps(b_ptr + i);
                _mm256_storeu_ps(out_ptr + i, op(va, vb));
            } else {
                size_t tail = n - i;
                __m256 va = masked_loadu_ps(a_ptr + i, tail);
                __m256 vb = masked_loadu_ps(b_ptr + i, tail);
                masked_storeu_ps(out_ptr + i, op(va, vb), tail);
            }
        }
    }
    return out;
}

template <typename Func>
Tensor binary_op_dispatch(const Tensor& A, const Tensor& B, Func op) {
    if (A.is_contiguous() && B.is_contiguous() && A.numel() == B.numel()) {
        return binary_op_contiguous(A, B, op);
    }
    return binary_op_general(A, B, op);
}

// ---------------------- Tiled Matrix Multiplication (GEMM) ----------------------

Tensor matmul_avx2_f32(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim != 2 || B.impl->ndim != 2) throw std::runtime_error("matmul: only 2D");
    size_t M = A.impl->shape[0];
    size_t K = A.impl->shape[1];
    size_t N = B.impl->shape[1];
    if (K != B.impl->shape[0]) throw std::runtime_error("matmul: K mismatch");

    Tensor A_contig = A.is_contiguous() ? A : A.contiguous();
    Tensor B_contig = B.is_contiguous() ? B : B.contiguous();
    
    Tensor C({M, N}, DType::Float32);
    
    const float* a_ptr = get_ptr<float>(A_contig);
    const float* b_ptr = get_ptr<float>(B_contig);
    float* c_ptr = get_ptr<float>(C);
    
    std::memset(c_ptr, 0, M * N * sizeof(float));

    // L1 Cache Tiling
    const size_t MC = 64;  
    const size_t KC = 32;
    const size_t NC = 128; 

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i0 = 0; i0 < M; i0 += MC) {
        for (size_t j0 = 0; j0 < N; j0 += NC) {
            
            size_t i_end = std::min(i0 + MC, M);
            size_t j_end = std::min(j0 + NC, N);
            
            for (size_t k0 = 0; k0 < K; k0 += KC) {
                size_t k_end = std::min(k0 + KC, K);

                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        __m256 va = _mm256_set1_ps(a_ptr[i * K + k]);
                        
                        size_t j = j0;
                        for (; j + 16 <= j_end; j += 16) {
                            float* c_loc = c_ptr + i * N + j;
                            const float* b_loc = b_ptr + k * N + j;
                            
                            __m256 vc1 = _mm256_loadu_ps(c_loc);
                            __m256 vb1 = _mm256_loadu_ps(b_loc);
                            vc1 = _mm256_fmadd_ps(va, vb1, vc1);
                            _mm256_storeu_ps(c_loc, vc1);
                            
                            __m256 vc2 = _mm256_loadu_ps(c_loc + 8);
                            __m256 vb2 = _mm256_loadu_ps(b_loc + 8);
                            vc2 = _mm256_fmadd_ps(va, vb2, vc2);
                            _mm256_storeu_ps(c_loc + 8, vc2);
                        }
                        for (; j < j_end; ++j) {
                            c_ptr[i * N + j] += a_ptr[i * K + k] * b_ptr[k * N + j];
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

// ---------------------- Implementations ----------------------

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
     return binary_op_dispatch(a, b, [](__m256 x, __m256 y){ 
         alignas(32) float ta[8], tb[8], tr[8];
         _mm256_store_ps(ta, x);
         _mm256_store_ps(tb, y);
         for(int i=0; i<8; ++i) tr[i] = std::pow(ta[i], tb[i]);
         return _mm256_load_ps(tr);
     });
}

// Comparisons (CMP_OP)
template<int OP>
Tensor cmp_op(const Tensor& a, const Tensor& b) {
    return binary_op_dispatch(a, b, []( __m256 x, __m256 y){
        return _mm256_and_ps(_mm256_cmp_ps(x, y, OP), YMM_1_PS);
    });
}

Tensor lt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_LT_OQ>(a,b); }
Tensor le_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_LE_OQ>(a,b); }
Tensor gt_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_GT_OQ>(a,b); }
Tensor ge_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_GE_OQ>(a,b); }
Tensor eq_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx2_f32(const Tensor& a, const Tensor& b) { return cmp_op<_CMP_NEQ_OQ>(a,b); }

// Unary
template <typename Func>
Tensor unary_op_simple(const Tensor& A, Func op) {
    Tensor out(A.shape(), DType::Float32);
    Tensor Ac = A.is_contiguous() ? A : A.contiguous();
    
    const float* in = get_ptr<float>(Ac);
    float* o = get_ptr<float>(out);
    size_t n = Ac.numel();

    if (n < AVX_OMP_THRESHOLD) {
        size_t i = 0;
        for (; i+8 <= n; i+=8) {
            _mm256_storeu_ps(o+i, op(_mm256_loadu_ps(in+i)));
        }
        for (; i<n; ++i) {
             alignas(32) float tmp[8];
             _mm256_store_ps(tmp, op(_mm256_set1_ps(in[i])));
             o[i] = tmp[0];
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i += 8) {
             size_t tail = std::min((size_t)8, n - i);
             __m256 v = masked_loadu_ps(in + i, tail);
             masked_storeu_ps(o + i, op(v), tail);
        }
    }
    return out;
}

Tensor abs_avx2_f32(const Tensor& a) { return unary_op_simple(a, [](__m256 x){ 
    return _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))); 
}); }

Tensor sqrt_avx2_f32(const Tensor& a) { return unary_op_simple(a, [](__m256 x){ return _mm256_sqrt_ps(x); }); }

// Fallbacks
#define SCALAR_FALLBACK_UNARY(NAME, FUNC) \
Tensor NAME##_avx2_f32(const Tensor& a) { \
    Tensor out(a.shape(), DType::Float32); \
    Tensor ac = a.contiguous(); \
    const float* iptr = get_ptr<float>(ac); \
    float* optr = get_ptr<float>(out); \
    size_t n = ac.numel(); \
    _Pragma("omp parallel for if(n > 32768)") \
    for(size_t i=0; i<n; ++i) optr[i] = FUNC(iptr[i]); \
    return out; \
}

SCALAR_FALLBACK_UNARY(ln, std::log)
SCALAR_FALLBACK_UNARY(exp, std::exp)
SCALAR_FALLBACK_UNARY(sin, std::sin)
SCALAR_FALLBACK_UNARY(cos, std::cos)
SCALAR_FALLBACK_UNARY(tan, std::tan)
SCALAR_FALLBACK_UNARY(asin, std::asin)
SCALAR_FALLBACK_UNARY(acos, std::acos)
SCALAR_FALLBACK_UNARY(atan, std::atan)
SCALAR_FALLBACK_UNARY(sinh, std::sinh)
SCALAR_FALLBACK_UNARY(cosh, std::cosh)
SCALAR_FALLBACK_UNARY(tanh, std::tanh)
SCALAR_FALLBACK_UNARY(sigmoid, [](float x){ return 1.0f/(1.0f+std::exp(-x)); })
SCALAR_FALLBACK_UNARY(softplus, [](float x){ return std::log(1.0f+std::exp(x)); })
Tensor relu_avx2_f32(const Tensor& a) { return unary_op_simple(a, [](__m256 x){ return _mm256_max_ps(x, YMM_0_PS); }); }

// Reductions (Sum)
Tensor sum_avx2_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("AVX sum: only dim=-1 supported");
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    size_t n = tc.numel();
    const float* data = get_ptr<float>(tc);
    float total = 0.0f;
    
    #pragma omp parallel reduction(+:total) if(n > AVX_OMP_THRESHOLD)
    {
        float local_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < n; i += 8) {
            if (i+8 <= n) {
                __m256 v = _mm256_loadu_ps(data + i);
                __m128 vlow = _mm256_castps256_ps128(v);
                __m128 vhigh = _mm256_extractf128_ps(v, 1);
                vlow = _mm_add_ps(vlow, vhigh);
                __m128 shuf = _mm_movehdup_ps(vlow);
                __m128 sums = _mm_add_ps(vlow, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                local_sum += _mm_cvtss_f32(sums);
            } else {
                for(size_t j=i; j<n; ++j) local_sum += data[j];
            }
        }
        total += local_sum;
    }
    
    Tensor out({1}, DType::Float32);
    ((float*)get_ptr<float>(out))[0] = total;
    return out;
}

Tensor mean_avx2_f32(const Tensor& t, int dim) {
    Tensor s = sum_avx2_f32(t, dim);
    float n = (float)t.numel();
    ((float*)get_ptr<float>(s))[0] /= n;
    return s;
}

Tensor max_avx2_f32(const Tensor& t, int dim) {
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    const float* data = get_ptr<float>(tc);
    float m = -std::numeric_limits<float>::infinity();
    for(size_t i=0; i<tc.numel(); ++i) if(data[i]>m) m = data[i];
    Tensor out({1}, DType::Float32); ((float*)get_ptr<float>(out))[0] = m; return out;
}
Tensor min_avx2_f32(const Tensor& t, int dim) {
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    const float* data = get_ptr<float>(tc);
    float m = std::numeric_limits<float>::infinity();
    for(size_t i=0; i<tc.numel(); ++i) if(data[i]<m) m = data[i];
    Tensor out({1}, DType::Float32); ((float*)get_ptr<float>(out))[0] = m; return out;
}

#endif