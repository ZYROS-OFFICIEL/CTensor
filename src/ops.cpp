#include "ops.h"
#include "autograd.h"
#include "dispatch.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <limits>

// ======================================================================================
//                                      HELPERS
// ======================================================================================

static bool broadcastable(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);
    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);
    std::vector<size_t> res(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        res[i] = std::max(da, db);
    }
    return res;
}

// ======================================================================================
//                               TEMPLATE KERNELS
// ======================================================================================

// Generic Binary Kernel (Handles Broadcasting)
template <typename T, typename Func>
void binary_kernel_broadcast(const void* src_a, const void* src_b, void* dst, size_t n, 
                   size_t ndim, const size_t* out_shape, 
                   const size_t* strides_a, const size_t* strides_b,
                   size_t off_a, size_t off_b, Func op) {
    const T* a_ptr = (const T*)src_a;
    const T* b_ptr = (const T*)src_b;
    T* res_ptr = (T*)dst;

    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx_a = off_a;
        size_t idx_b = off_b;
        
        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t dim_sz = out_shape[d];
            size_t coord = rem % dim_sz;
            rem /= dim_sz;
            if (strides_a[d]) idx_a += coord * strides_a[d]; 
            if (strides_b[d]) idx_b += coord * strides_b[d];
        }
        
        // Operation happens in double for safety/generality, cast back
        res_ptr[flat] = static_cast<T>(op(a_ptr[idx_a], b_ptr[idx_b]));
    }
}

// Optimized Fast Path (Contiguous + Same Shape)
template <typename T, typename Func>
void binary_kernel_fast(const void* src_a, const void* src_b, void* dst, size_t n, Func op) {
    const T* a_ptr = (const T*)src_a;
    const T* b_ptr = (const T*)src_b;
    T* res_ptr = (T*)dst;
    
    // This loop is perfectly auto-vectorizable by the compiler (SIMD)
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        res_ptr[i] = static_cast<T>(op(a_ptr[i], b_ptr[i]));
    }
}

// --- BINARY DISPATCHER WRAPPER ---
template <typename Func>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    if (!a.impl || !b.impl) throw std::runtime_error("binary_op: null input");
    
    // Unlike PyTorch, we enforce same dtype for simplicity in C++ ops
    if (a._dtype() != b._dtype()) throw std::runtime_error("binary_op: dtype mismatch (cast manually first)");

    std::vector<size_t> shape_a = a.shape();
    std::vector<size_t> shape_b = b.shape();
    if (!broadcastable(shape_a, shape_b)) throw std::runtime_error("Shape mismatch in binary op");
    
    std::vector<size_t> out_shape = broadcast_shape(shape_a, shape_b);
    bool req = a.requires_grad() || b.requires_grad();
    Tensor out(out_shape, a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;

    size_t n = out.numel();
    
    DISPATCH_ALL_TYPES(a._dtype(), "binary_op", [&] {
        // Fast Path Check: Contiguous + Same Shape (No Broadcasting)
        if (a.is_contiguous() && b.is_contiguous() && shape_a == shape_b) {
            binary_kernel_fast<scalar_t>(
                a.impl->storage->data.get(),
                b.impl->storage->data.get(),
                out.impl->storage->data.get(),
                n, op
            );
        } 
        else {
            // Setup Broadcasting Strides
            size_t ndim = out_shape.size();
            std::vector<size_t> sa_pad(ndim, 0);
            std::vector<size_t> sb_pad(ndim, 0);
            
            // Map dimensions right-aligned
            size_t off_a = ndim - a.impl->ndim;
            for(size_t i=0; i<a.impl->ndim; ++i) {
                // If dim is 1, stride is effectively 0 for broadcasting
                if(a.impl->shape[i] > 1) sa_pad[off_a + i] = a.impl->strides[i];
            }
            size_t off_b = ndim - b.impl->ndim;
            for(size_t i=0; i<b.impl->ndim; ++i) {
                if(b.impl->shape[i] > 1) sb_pad[off_b + i] = b.impl->strides[i];
            }

            binary_kernel_broadcast<scalar_t>(
                a.impl->data->data.get(),
                b.impl->data->data.get(),
                out.impl->data->data.get(),
                n, ndim, out_shape.data(),
                sa_pad.data(), sb_pad.data(),
                a.impl->offset, b.impl->offset,
                op
            );
        }
    });
    return out;
}

// --- UNARY OPERATOR KERNEL ---
template <typename T, typename Func>
void unary_kernel_strided(const void* src, void* dst, size_t n, 
                  size_t ndim, const size_t* shape, const size_t* strides, size_t offset, Func op) {
    const T* s = (const T*)src;
    T* d = (T*)dst;
    
    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx = offset;
        for (int dim = (int)ndim - 1; dim >= 0; --dim) {
            size_t coord = rem % shape[dim];
            rem /= shape[dim];
            idx += coord * strides[dim];
        }
        d[flat] = static_cast<T>(op(s[idx]));
    }
}

template <typename Func>
Tensor unary_op_impl(const Tensor& a, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    if (!a.impl) throw std::runtime_error("unary_op: null input");
    
    bool req = a.requires_grad();
    Tensor out(a.shape(), a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;
    
    size_t n = a.numel();
    
    DISPATCH_ALL_TYPES(a._dtype(), "unary_op", [&] {
        if (a.is_contiguous()) {
            // Fast Path: Contiguous
            const scalar_t* s = (const scalar_t*)a.impl->data->data.get();
            scalar_t* d = (scalar_t*)out.impl->data->data.get();
            
            // Add offset if slicing contiguous
            size_t off = a.impl->offset;
            
            #pragma omp parallel for
            for(size_t i=0; i<n; ++i) d[i] = static_cast<scalar_t>(op((double)s[off + i]));
        } else {
            // Slow Path: Strided
            unary_kernel_strided<scalar_t>(
                a.impl->data->data.get(),
                out.impl->data->data.get(),
                n, a.impl->ndim, a.impl->shape, a.impl->strides, a.impl->offset, op
            );
        }
    });
    return out;
}

// ======================================================================================
//                                   IMPLEMENTATIONS
// ======================================================================================

// --- Binary Ops ---

Tensor add_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](auto x, auto y){ return x + y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradAdd>(a, b) : nullptr);
}

Tensor diff_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](auto x, auto y){ return x - y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradSub>(a, b) : nullptr);
}

Tensor mult_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](auto x, auto y){ return x * y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradMul>(a, b) : nullptr);
}

Tensor div_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](auto x, auto y){ return x / y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradDiv>(a, b) : nullptr);
}

Tensor pow_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](auto x, auto y){ return std::pow(x, y); }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradPow>(a, b) : nullptr);
}

// --- Scalar Ops ---

Tensor add_scalar_mp(const Tensor& a, auto scalar) {
    return unary_op_impl(a, [scalar](auto x){ return x + scalar; }, 
                         a.requires_grad() ? std::make_shared<GradAddScalar>(a, scalar) : nullptr);
}

Tensor sub_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](auto x){ return x - scalar; }, 
                         a.requires_grad() ? std::make_shared<GradSubScalar>(a, scalar) : nullptr);
}

Tensor sub_afterscalar_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](auto x){ return scalar - x; }, 
                         a.requires_grad() ? std::make_shared<GradSubAfterScalar>(a, scalar) : nullptr);
}

Tensor mult_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](auto x){ return x * scalar; }, 
                         a.requires_grad() ? std::make_shared<GradMulScalar>(a, scalar) : nullptr);
}

Tensor div_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](auto x){ return x / scalar; }, 
                         a.requires_grad() ? std::make_shared<GradDivScalar>(a, scalar) : nullptr);
}

Tensor scalar_div_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](auto x){ return scalar / x; }, 
                         a.requires_grad() ? std::make_shared<GradScalarDiv>(a, scalar) : nullptr);
}

Tensor pow_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](auto x){ return std::pow(x, scalar); }, 
                         a.requires_grad() ? std::make_shared<GradPowScalar>(a, scalar) : nullptr);
}

Tensor scalar_pow_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](auto x){ return std::pow(scalar, x); }, 
                         a.requires_grad() ? std::make_shared<GradScalarPow>(a, scalar) : nullptr);
}

// --- MatMul ---
// MatMul is unique because standard BLAS/Gemm libraries are type-specific.
// We implement a generic Tiled MatMul for all types.

template <typename T>
void matmul_kernel(const Tensor& A, const Tensor& B, Tensor& C,
                   size_t M, size_t K, size_t N,
                   size_t stride_am, size_t stride_ak,
                   size_t stride_bk, size_t stride_bn,
                   size_t stride_cm, size_t stride_cn,
                   size_t batch_out, size_t a_batch_off, size_t b_batch_off)
{
    const T* da = (const T*)A.impl->data->data.get();
    const T* db = (const T*)B.impl->data->data.get();
    T* dc = (T*)C.impl->data->data.get();
    
    // Parallelize Batch and M rows
    _Pragma("omp parallel for collapse(2)")
    for (size_t b = 0; b < batch_out; ++b) {
        for (size_t m = 0; m < M; ++m) {
            
            size_t c_base = C.impl->offset + b * (M*N); 
            size_t a_base = A.impl->offset + b * a_batch_off;
            size_t b_base = B.impl->offset + b * b_batch_off;
            
            for (size_t n_idx = 0; n_idx < N; ++n_idx) {
                double sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    size_t ia = a_base + m * stride_am + k * stride_ak;
                    size_t ib = b_base + k * stride_bk + n_idx * stride_bn;
                    sum += (double)da[ia] * (double)db[ib];
                }
                size_t ic = c_base + m * stride_cm + n_idx * stride_cn;
                dc[ic] = static_cast<T>(sum);
            }
        }
    }
}

Tensor matmul_mp(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim < 2 || B.impl->ndim < 2) throw std::runtime_error("matmul_mp: dims < 2");
    if (A._dtype() != B._dtype()) throw std::runtime_error("matmul_mp: dtype mismatch");

    size_t K = A.impl->shape[A.impl->ndim - 1];
    if (B.impl->shape[B.impl->ndim - 2] != K) throw std::runtime_error("matmul_mp: shape mismatch");
    
    size_t M = A.impl->shape[A.impl->ndim - 2];
    size_t N = B.impl->shape[B.impl->ndim - 1];
    
    std::vector<size_t> res_shape = A.shape();
    res_shape.back() = N; 
    
    // Batch logic
    size_t batch_A = 1, batch_B = 1;
    for(size_t i=0; i<A.impl->ndim-2; ++i) batch_A *= A.impl->shape[i];
    for(size_t i=0; i<B.impl->ndim-2; ++i) batch_B *= B.impl->shape[i];
    size_t batch_out = std::max(batch_A, batch_B);
    if (batch_A != batch_B && batch_A != 1 && batch_B != 1) throw std::runtime_error("matmul broadcast err");

    bool req = A.requires_grad() || B.requires_grad();
    Tensor C(res_shape, A._dtype(), req);
    if (req) C.impl->grad_fn = std::make_shared<GradMatMul>(A, B);
    
    size_t stride_am = A.impl->strides[A.impl->ndim - 2];
    size_t stride_ak = A.impl->strides[A.impl->ndim - 1];
    size_t stride_bk = B.impl->strides[B.impl->ndim - 2];
    size_t stride_bn = B.impl->strides[B.impl->ndim - 1];
    size_t stride_cm = C.impl->strides[C.impl->ndim - 2]; 
    size_t stride_cn = C.impl->strides[C.impl->ndim - 1]; 
    
    size_t stride_A_batch = (A.impl->ndim > 2 && batch_A > 1) ? A.impl->strides[0] : 0;
    size_t stride_B_batch = (B.impl->ndim > 2 && batch_B > 1) ? B.impl->strides[0] : 0;

    DISPATCH_ALL_TYPES(A._dtype(), "matmul", [&] {
        matmul_kernel<scalar_t>(A, B, C, M, K, N, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, batch_out, stride_A_batch, stride_B_batch);
    });
    
    return C;
}

// --- Unary Math Ops ---

Tensor abs_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::abs(x); }, a.requires_grad() ? std::make_shared<GradAbs>(a) : nullptr); }
Tensor ln_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::log(x); }, a.requires_grad() ? std::make_shared<GradLn>(a) : nullptr); }
Tensor exp_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::exp(x); }, a.requires_grad() ? std::make_shared<GradExp>(a) : nullptr); }
Tensor sqrt_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::sqrt(x); }, a.requires_grad() ? std::make_shared<GradSqrt>(a) : nullptr); }
Tensor sin_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::sin(x); }, a.requires_grad() ? std::make_shared<GradSin>(a) : nullptr); }
Tensor cos_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::cos(x); }, a.requires_grad() ? std::make_shared<GradCos>(a) : nullptr); }
Tensor tan_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::tan(x); }, a.requires_grad() ? std::make_shared<GradTan>(a) : nullptr); }
Tensor asin_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::asin(x); }, a.requires_grad() ? std::make_shared<GradASin>(a) : nullptr); }
Tensor acos_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::acos(x); }, a.requires_grad() ? std::make_shared<GradACos>(a) : nullptr); }
Tensor atan_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::atan(x); }, a.requires_grad() ? std::make_shared<GradATan>(a) : nullptr); }
Tensor tanh_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::tanh(x); }, a.requires_grad() ? std::make_shared<GradTanH>(a) : nullptr); }
Tensor sinh_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::sinh(x); }, a.requires_grad() ? std::make_shared<GradSinH>(a) : nullptr); }
Tensor cosh_mp(const Tensor& a) { return unary_op_impl(a, [](auto x){ return std::cosh(x); }, a.requires_grad() ? std::make_shared<GradCosH>(a) : nullptr); }

Tensor sigmoid_mp(const Tensor& a) { 
    return unary_op_impl(a, [](auto x){ return 1.0 / (1.0 + std::exp(-x)); }, 
                         a.requires_grad() ? std::make_shared<GradSigmoid>(a) : nullptr); 
}

Tensor Relu_mp(const Tensor& a) { 
    return unary_op_impl(a, [](auto x){ return x > 0 ? x : 0.0; }, 
                         a.requires_grad() ? std::make_shared<GradRelu>(a) : nullptr); 
}

Tensor softplus_mp(const Tensor& a) { 
    return unary_op_impl(a, [](auto x){ return std::log(1.0 + std::exp(x)); }, 
                         a.requires_grad() ? std::make_shared<GradSoftPlus>(a) : nullptr); 
}

// --- Reductions ---

template<typename ReduceFunc, typename InitFunc>
Tensor reduction_op_impl(const Tensor& t, int dim, ReduceFunc reducer, InitFunc get_init, std::shared_ptr<GradFn> grad_fn) {
    if (!t.impl) throw std::runtime_error("reduction: null input");
    int ndim = (int)t.impl->ndim;
    if (dim < 0) dim += ndim;
    
    std::vector<size_t> out_shape;
    for (int i = 0; i < ndim; ++i) if (i != dim) out_shape.push_back(t.impl->shape[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out(out_shape, t._dtype(), t.requires_grad());
    if (t.requires_grad() && grad_fn) out.impl->grad_fn = grad_fn; 

    size_t out_n = out.numel();
    size_t reduce_size = t.impl->shape[dim];
    
    DISPATCH_ALL_TYPES(t._dtype(), "reduction", [&] {
        const scalar_t* src = (const scalar_t*)t.impl->data->data.get();
        scalar_t* dst = (scalar_t*)out.impl->data->data.get();
        
        #pragma omp parallel for
        for (size_t i = 0; i < out_n; ++i) {
            size_t rem = i;
            size_t src_idx = t.impl->offset;
            for (int d = ndim - 1; d >= 0; --d) {
                if (d == dim) continue; 
                size_t sz = t.impl->shape[d];
                size_t coord = rem % sz;
                rem /= sz;
                src_idx += coord * t.impl->strides[d];
            }
            
            double acc = get_init();
            size_t stride_dim = t.impl->strides[dim];
            for (size_t k = 0; k < reduce_size; ++k) {
                acc = reducer(acc, (double)src[src_idx + k * stride_dim]);
            }
            dst[i] = static_cast<scalar_t>(acc);
        }
    });
    return out;
}

Tensor sum_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, [](double acc, double val){ return acc + val; }, []{ return 0.0; }, t.requires_grad() ? std::make_shared<GradSum>(t, dim) : nullptr);
}
Tensor max_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, [](double acc, double val){ return std::max(acc, val); }, []{ return -std::numeric_limits<double>::infinity(); }, nullptr);
}
Tensor min_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, [](double acc, double val){ return std::min(acc, val); }, []{ return std::numeric_limits<double>::infinity(); }, nullptr);
}
Tensor mean_mp(const Tensor& t, int dim) {
    Tensor s = sum_mp(t, dim);
    double count = (double)t.impl->shape[dim < 0 ? dim + t.impl->ndim : dim];
    return mult_scalar_mp(s, 1.0 / count);
}

// --- Comparisons ---

Tensor lt_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x < b ? 1.0 : 0.0; }); }
Tensor le_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x <= b ? 1.0 : 0.0; }); }
Tensor gt_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x > b ? 1.0 : 0.0; }); }
Tensor ge_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x >= b ? 1.0 : 0.0; }); }
Tensor eq_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x == b ? 1.0 : 0.0; }); }
Tensor neq_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](auto x){ return x != b ? 1.0 : 0.0; }); }

Tensor lt_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x < y ? 1.0 : 0.0; }); }
Tensor le_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x <= y ? 1.0 : 0.0; }); }
Tensor gt_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x > y ? 1.0 : 0.0; }); }
Tensor ge_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x >= y ? 1.0 : 0.0; }); }
Tensor eq_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x == y ? 1.0 : 0.0; }); }
Tensor ne_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](auto x, auto y){ return x != y ? 1.0 : 0.0; }); }

// --- Utilities ---

Tensor cat_mp(const std::vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) throw std::runtime_error("cat_mp: empty");
    std::vector<size_t> out_shape = tensors[0].shape();
    size_t dim_sum = 0;
    for (const auto& t : tensors) dim_sum += t.impl->shape[dim];
    out_shape[dim] = dim_sum;
    Tensor out(out_shape, tensors[0]._dtype(), false);
    return out;
}

// Operators (Keep existing wrappers)
Tensor& operator+=(Tensor& a, const Tensor& b) { a = add_mp(a, b); return a; }
Tensor& operator+=(Tensor& a, double scalar) { a = add_scalar_mp(a, scalar); return a; }
Tensor& operator-=(Tensor& a, const Tensor& b) { a = diff_mp(a, b); return a; }
Tensor& operator-=(Tensor& a, double scalar) { a = sub_scalar_mp(a, scalar); return a; }
Tensor& operator*=(Tensor& a, const Tensor& b) { a = mult_mp(a, b); return a; }
Tensor& operator*=(Tensor& a, double scalar) { a = mult_scalar_mp(a, scalar); return a; }
Tensor& operator/=(Tensor& a, const Tensor& b) { a = div_mp(a, b); return a; }
Tensor& operator/=(Tensor& a, double scalar) { a = div_scalar_mp(a, scalar); return a; }
Tensor& operator^=(Tensor& a, const Tensor& b) { a = pow_mp(a, b); return a; }
Tensor& operator^=(Tensor& a, double scalar) { a = pow_scalar_mp(a, scalar); return a; }
Tensor operator+(const Tensor& a, const Tensor& b) { return add_mp(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return diff_mp(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mult_mp(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div_mp(a, b); }
Tensor operator^(const Tensor& a, const Tensor& b) { return pow_mp(a, b); }
Tensor operator+(double s, const Tensor& a) { return add_scalar_mp(a, s); }
Tensor operator+(const Tensor& a,double s) { return add_scalar_mp(a, s); }