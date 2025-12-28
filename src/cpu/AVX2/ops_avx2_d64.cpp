#include "ops_avx2_d64.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>
#include <cstring>
#include <functional>

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
//                     Internal AVX2 Double Constants & Helpers
// ========================================================================
#define ZMM_1_PD  _mm256_set1_pd(1.0)
#define ZMM_0_PD  _mm256_setzero_pd()
#define ZMM_INF_PD _mm256_set1_pd(std::numeric_limits<double>::infinity())
#define ZMM_NEG_INF_PD _mm256_set1_pd(-std::numeric_limits<double>::infinity())

// --- Masked Load/Store Helpers (Manual Stack Buffer) ---
// Safer near page boundaries and cleaner API than intrinsics for simple tails.

static inline __m256d masked_loadu_pd(const double* ptr, size_t valid_count) {
    alignas(32) double tmp[4] = {0.0};
    for (size_t i = 0; i < valid_count; ++i) tmp[i] = ptr[i];
    return _mm256_load_pd(tmp);
}

static inline void masked_storeu_pd(double* ptr, __m256d v, size_t valid_count) {
    alignas(32) double tmp[4];
    _mm256_store_ps((float*)tmp, _mm256_castpd_ps(v)); // cast to ps for store, data is same
    // Standard store
    _mm256_store_pd(tmp, v);
    for (size_t i = 0; i < valid_count; ++i) ptr[i] = tmp[i];
}

// --- Horizontal Sum ---
inline double hsum256_pd(__m256d v) {
    // v = [d3, d2, d1, d0]
    __m256d v2 = _mm256_permute2f128_pd(v, v, 1); // [d1, d0, d3, d2]
    v = _mm256_add_pd(v, v2);                     // [d3+d1, d2+d0, d1+d3, d0+d2]
    __m256d v3 = _mm256_permute_pd(v, 0x5);       // shuffle to swap pairs: 0b0101
    v = _mm256_add_pd(v, v3);
    return _mm256_cvtsd_f64(v);
}

} // namespace

// ---------------------------
// Broadcasting helpers
// ---------------------------

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

// ----------------------Binary Broadcast Template (Double)----------------------

template <typename Func>
Tensor binary_op_broadcast_d64(const Tensor& A, const Tensor& B, Func op) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    // FIX: Removed A.device()
    Tensor out(out_shape, DType::Double64);

    const double* a_ptr = get_ptr<double>(A);
    const double* b_ptr = get_ptr<double>(B);
    double* out_ptr = get_ptr<double>(out);

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
        int32_t gather_idx[4]; 

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
                // _mm256_i32gather_pd requires 128-bit integer index vector
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
            __m256d vr = op(va, vb);
            _mm256_storeu_pd(out_ptr + i, vr);
        }
    }

    // Tail
    if (vec_end < out_numel) {
        size_t tail_count = out_numel - vec_end;
        
        // Tail Load A
        __m256d va_tail;
        if (a_contig) {
            va_tail = masked_loadu_pd(a_ptr + vec_end, tail_count);
        } else if (a_is_scalar) {
            va_tail = _mm256_set1_pd(a_ptr[0]);
        } else {
            alignas(32) double tmp[4];
            for(size_t j=0; j<tail_count; ++j) {
                size_t offset = compute_offset_bytes(vec_end + j, out_shape, out_mult, a_shape, a_strides);
                tmp[j] = *(const double*)((const char*)a_ptr + offset);
            }
            va_tail = _mm256_load_pd(tmp);
        }

        // Tail Load B
        __m256d vb_tail;
        if (b_contig) {
            vb_tail = masked_loadu_pd(b_ptr + vec_end, tail_count);
        } else if (b_is_scalar) {
            vb_tail = _mm256_set1_pd(b_ptr[0]);
        } else {
            alignas(32) double tmp[4];
            for(size_t j=0; j<tail_count; ++j) {
                size_t offset = compute_offset_bytes(vec_end + j, out_shape, out_mult, b_shape, b_strides);
                tmp[j] = *(const double*)((const char*)b_ptr + offset);
            }
            vb_tail = _mm256_load_pd(tmp);
        }

        __m256d vr_tail = op(va_tail, vb_tail);
        masked_storeu_pd(out_ptr + vec_end, vr_tail, tail_count);
    }

    return out;
}

// Unary Template (Double)
template <typename Func>
Tensor unary_op_d64_impl(const Tensor& A, Func op) {
    Tensor out(A.shape(), DType::Double64);
    const double* a_ptr = get_ptr<double>(A);
    double* out_ptr = get_ptr<double>(out);
    size_t n = A.numel();
    size_t vec_end = (n / 4) * 4;

    #pragma omp parallel for
    for (size_t i = 0; i < vec_end; i += 4) {
        __m256d va = _mm256_loadu_pd(a_ptr + i);
        __m256d vr = op(va);
        _mm256_storeu_pd(out_ptr + i, vr);
    }

    if (vec_end < n) {
        size_t tail = n - vec_end;
        __m256d va = masked_loadu_pd(a_ptr + vec_end, tail);
        __m256d vr = op(va);
        masked_storeu_pd(out_ptr + vec_end, vr, tail);
    }
    return out;
}


/*------------------------------------------Binary API---------------------------------------------------*/

Tensor add_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_d64(a, b, [](__m256d x, __m256d y){ return _mm256_add_pd(x, y); });
}
Tensor sub_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_d64(a, b, [](__m256d x, __m256d y){ return _mm256_sub_pd(x, y); });
}
Tensor mul_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_d64(a, b, [](__m256d x, __m256d y){ return _mm256_mul_pd(x, y); });
}
Tensor div_avx2_d64(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_d64(a, b, [](__m256d x, __m256d y){ return _mm256_div_pd(x, y); });
}

// Pow: Use OMP SIMD instead of raw AVX for double precision power (no AVX2 intrinsic for pow)
Tensor pow_avx2_d64(const Tensor& A, const Tensor& B) {
    std::vector<size_t> a_shape = A.shape();
    std::vector<size_t> b_shape = B.shape();
    std::vector<size_t> out_shape = broadcast_shape(a_shape, b_shape);
    size_t out_numel = 1; for (auto s : out_shape) out_numel *= s;
    Tensor out(out_shape, DType::Double64);

    const double* a_ptr = get_ptr<double>(A);
    const double* b_ptr = get_ptr<double>(B);
    double* out_ptr = get_ptr<double>(out);

    bool a_contig = (A.is_contiguous()) && (a_shape == out_shape);
    bool b_contig = (B.is_contiguous()) && (b_shape == out_shape);

    if (a_contig && b_contig) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < out_numel; ++i) {
            out_ptr[i] = std::pow(a_ptr[i], b_ptr[i]);
        }
    } else {
        auto out_mult = build_index_multipliers(out_shape);
        auto a_strides = shape_to_strides_bytes(a_shape);
        auto b_strides = shape_to_strides_bytes(b_shape);

        #pragma omp parallel for
        for (size_t i = 0; i < out_numel; ++i) {
            int32_t a_off = compute_offset_bytes(i, out_shape, out_mult, a_shape, a_strides);
            int32_t b_off = compute_offset_bytes(i, out_shape, out_mult, b_shape, b_strides);
            double val_a = *(const double*)((const char*)a_ptr + a_off);
            double val_b = *(const double*)((const char*)b_ptr + b_off);
            out_ptr[i] = std::pow(val_a, val_b);
        }
    }
    return out;
}

/*----------------------Comparisons---------------------------*/

template<int CMP_FLAG>
Tensor cmp_avx2_d64_impl(const Tensor& a, const Tensor& b) {
    return binary_op_broadcast_d64(a, b, []( __m256d x, __m256d y){
        __m256d m = _mm256_cmp_pd(x, y, CMP_FLAG);
        // mask bits are 0xFFFFFFFFFFFFFFFF for true.
        // AND with 1.0 to get 1.0 for true, 0.0 for false.
        return _mm256_and_pd(m, ZMM_1_PD);
    });
}

Tensor lt_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_LT_OQ>(a,b); }
Tensor le_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_LE_OQ>(a,b); }
Tensor gt_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_GT_OQ>(a,b); }
Tensor ge_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_GE_OQ>(a,b); }
Tensor eq_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_EQ_OQ>(a,b); }
Tensor ne_avx2_d64(const Tensor& a, const Tensor& b) { return cmp_avx2_d64_impl<_CMP_NEQ_OQ>(a,b); }


/*----------------------Matmul (Double)---------------------------*/

Tensor matmul_avx2_d64(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2) throw std::runtime_error("matmul_avx2_d64: only 2D tensors");
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];
    if (K != B.shape()[0]) throw std::runtime_error("matmul_avx2_d64: shape mismatch");

    Tensor C({M, N}, DType::Double64);
    const double* a_ptr = get_ptr<double>(A);
    const double* b_ptr = get_ptr<double>(B);
    double* c_ptr = get_ptr<double>(C);
    
    std::memset(c_ptr, 0, M * N * sizeof(double));

    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            double a_val = a_ptr[i * K + k];
            __m256d va = _mm256_set1_pd(a_val);
            
            size_t j = 0;
            // Unroll 4 doubles at a time
            for (; j + 4 <= N; j += 4) {
                double* c_addr = c_ptr + i * N + j;
                const double* b_addr = b_ptr + k * N + j;
                
                __m256d vc = _mm256_loadu_pd(c_addr);
                __m256d vb = _mm256_loadu_pd(b_addr);
                __m256d vres = _mm256_fmadd_pd(va, vb, vc);
                _mm256_storeu_pd(c_addr, vres);
            }
            // Tail
            if (j < N) {
                size_t tail = N - j;
                __m256d vc = masked_loadu_pd(c_ptr + i*N + j, tail);
                __m256d vb = masked_loadu_pd(b_ptr + k*N + j, tail);
                __m256d vres = _mm256_fmadd_pd(va, vb, vc);
                masked_storeu_pd(c_ptr + i*N + j, vres, tail);
            }
        }
    }
    return C;
}

/*----------------------Reductions (Double)---------------------------*/
Tensor sum_avx2_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_avx2_d64: only dim=-1");
    size_t n = t.numel();
    const double* data = get_ptr<double>(t);
    double global_sum = 0.0;
    size_t vec_end = (n/4)*4;

    #pragma omp parallel
    {
        __m256d vsum = _mm256_setzero_pd();
        #pragma omp for nowait
        for (size_t i=0; i<vec_end; i+=4) {
            vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(data+i));
        }
        double local_sum = hsum256_pd(vsum);
        #pragma omp atomic
        global_sum += local_sum;
    }
    for (size_t i=vec_end; i<n; ++i) global_sum += data[i];

    Tensor out({1}, DType::Double64);
    ((double*)get_ptr<double>(out))[0] = global_sum;
    return out;
}

Tensor mean_avx2_d64(const Tensor& t, int dim) {
    Tensor s = sum_avx2_d64(t, dim);
    double n = static_cast<double>(t.numel());
    ((double*)get_ptr<double>(s))[0] /= n;
    return s;
}

Tensor max_avx2_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_avx2_d64: only dim=-1");
    const double* data = get_ptr<double>(t);
    size_t n = t.numel();
    double global_max = -std::numeric_limits<double>::infinity();
    size_t vec_end = (n/4)*4;

    #pragma omp parallel
    {
        __m256d vmax = ZMM_NEG_INF_PD;
        #pragma omp for nowait
        for (size_t i=0; i<vec_end; i+=4) {
            vmax = _mm256_max_pd(vmax, _mm256_loadu_pd(data+i));
        }
        
        // Horizontal max
        __m256d y = _mm256_permute2f128_pd(vmax, vmax, 1);
        __m256d m1 = _mm256_max_pd(vmax, y);
        __m256d m2 = _mm256_permute_pd(m1, 0x5);
        __m256d m3 = _mm256_max_pd(m1, m2);
        double local_max = _mm256_cvtsd_f64(m3);

        #pragma omp critical
        {
            if (local_max > global_max) global_max = local_max;
        }
    }
    for (size_t i=vec_end; i<n; ++i) {
        if (data[i] > global_max) global_max = data[i];
    }

    Tensor out({1}, DType::Double64);
    ((double*)get_ptr<double>(out))[0] = global_max;
    return out;
}

Tensor min_avx2_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_avx2_d64: only dim=-1");
    const double* data = get_ptr<double>(t);
    size_t n = t.numel();
    double global_min = std::numeric_limits<double>::infinity();
    size_t vec_end = (n/4)*4;

    #pragma omp parallel
    {
        __m256d vmin = ZMM_INF_PD;
        #pragma omp for nowait
        for (size_t i=0; i<vec_end; i+=4) {
            vmin = _mm256_min_pd(vmin, _mm256_loadu_pd(data+i));
        }
        
        __m256d y = _mm256_permute2f128_pd(vmin, vmin, 1);
        __m256d m1 = _mm256_min_pd(vmin, y);
        __m256d m2 = _mm256_permute_pd(m1, 0x5);
        __m256d m3 = _mm256_min_pd(m1, m2);
        double local_min = _mm256_cvtsd_f64(m3);

        #pragma omp critical
        {
            if (local_min < global_min) global_min = local_min;
        }
    }
    for (size_t i=vec_end; i<n; ++i) {
        if (data[i] < global_min) global_min = data[i];
    }

    Tensor out({1}, DType::Double64);
    ((double*)get_ptr<double>(out))[0] = global_min;
    return out;
}

/*------------------------------------------Unary ops---------------------------------------------------*/

// AVX2 implementations for simple unary ops
Tensor abs_avx2_d64(const Tensor& a) {
    static const __m256i abs_mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return unary_op_d64_impl(a, []( __m256d x){
        return _mm256_and_pd(x, _mm256_castsi256_pd(abs_mask));
    });
}

Tensor sqrt_avx2_d64(const Tensor& a) {
    return unary_op_d64_impl(a, []( __m256d x){ return _mm256_sqrt_pd(x); });
}

Tensor relu_avx2_d64(const Tensor& a) {
    return unary_op_d64_impl(a, []( __m256d x){ return _mm256_max_pd(x, ZMM_0_PD); });
}

// Complex unary ops use OMP SIMD
#define OMP_SIMD_UNARY_D64(FUNC_NAME, STD_FUNC) \
Tensor FUNC_NAME(const Tensor& a) { \
    Tensor out(a.shape(), DType::Double64); \
    const double* pa = get_ptr<double>(a); \
    double* pout = get_ptr<double>(out); \
    size_t n = a.numel(); \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < n; ++i) { \
        pout[i] = STD_FUNC(pa[i]); \
    } \
    return out; \
}

OMP_SIMD_UNARY_D64(ln_avx2_d64, std::log)
OMP_SIMD_UNARY_D64(exp_avx2_d64, std::exp)
OMP_SIMD_UNARY_D64(sin_avx2_d64, std::sin)
OMP_SIMD_UNARY_D64(cos_avx2_d64, std::cos)
OMP_SIMD_UNARY_D64(tan_avx2_d64, std::tan)
OMP_SIMD_UNARY_D64(asin_avx2_d64, std::asin)
OMP_SIMD_UNARY_D64(acos_avx2_d64, std::acos)
OMP_SIMD_UNARY_D64(atan_avx2_d64, std::atan)
OMP_SIMD_UNARY_D64(sinh_avx2_d64, std::sinh)
OMP_SIMD_UNARY_D64(cosh_avx2_d64, std::cosh)
OMP_SIMD_UNARY_D64(tanh_avx2_d64, std::tanh)

// Sigmoid / Softplus helpers
inline double sigmoid_scalar(double x) { return 1.0 / (1.0 + std::exp(-x)); }
OMP_SIMD_UNARY_D64(sigmoid_avx2_d64, sigmoid_scalar)

inline double softplus_scalar(double x) { return std::log(1.0 + std::exp(x)); }
OMP_SIMD_UNARY_D64(softplus_avx2_d64, softplus_scalar)

#endif // __AVX2__