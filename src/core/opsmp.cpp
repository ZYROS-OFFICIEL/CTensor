#include "opsmp.h"
#include "autograd.h"
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

// --- BINARY OPERATOR KERNEL ---
template <typename Func>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    if (!a.impl || !b.impl) throw std::runtime_error("binary_op: null input");
    
    std::vector<size_t> shape_a = a.shape();
    std::vector<size_t> shape_b = b.shape();
    if (!broadcastable(shape_a, shape_b)) throw std::runtime_error("Shape mismatch in binary op");
    
    std::vector<size_t> out_shape = broadcast_shape(shape_a, shape_b);
    size_t ndim = out_shape.size();
    
    bool req = a.requires_grad() || b.requires_grad();
    Tensor out(out_shape, a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;

    size_t n = out.numel();
    // FIX: use .data() for vector access
    const size_t* out_shape_ptr = out.impl->shape.data();
    
    std::vector<size_t> strides_a_pad(ndim, 0); 
    std::vector<size_t> shape_a_pad(ndim, 1);
    std::vector<size_t> strides_b_pad(ndim, 0);
    std::vector<size_t> shape_b_pad(ndim, 1);
    
    size_t offset_a = a.impl->ndim < ndim ? ndim - a.impl->ndim : 0;
    for(size_t i=0; i<a.impl->ndim; ++i) {
        strides_a_pad[offset_a + i] = a.impl->strides[i];
        shape_a_pad[offset_a + i] = a.impl->shape[i];
    }
    
    size_t offset_b = b.impl->ndim < ndim ? ndim - b.impl->ndim : 0;
    for(size_t i=0; i<b.impl->ndim; ++i) {
        strides_b_pad[offset_b + i] = b.impl->strides[i];
        shape_b_pad[offset_b + i] = b.impl->shape[i];
    }

    const size_t* sa_ptr = strides_a_pad.data();
    const size_t* sha_ptr = shape_a_pad.data();
    const size_t* sb_ptr = strides_b_pad.data();
    const size_t* shb_ptr = shape_b_pad.data();
    
    size_t off_a_base = a.impl->offset;
    size_t off_b_base = b.impl->offset;
    
    if (a._dtype() == DType::Float32) {
        // FIX: impl->data, not impl->storage
        float* data_a = (float*)a.impl->data->data.get();
        float* data_b = (float*)b.impl->data->data.get();
        float* data_out = (float*)out.impl->data->data.get();
        
        #pragma omp parallel for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            size_t idx_a = off_a_base;
            size_t idx_b = off_b_base;
            
            for (int d = (int)ndim - 1; d >= 0; --d) {
                size_t dim_sz = out_shape_ptr[d];
                size_t coord = rem % dim_sz;
                rem /= dim_sz;
                if (sha_ptr[d] > 1) idx_a += coord * sa_ptr[d];
                if (shb_ptr[d] > 1) idx_b += coord * sb_ptr[d];
            }
            data_out[flat] = (float)op((double)data_a[idx_a], (double)data_b[idx_b]);
        }
    } else {
        #pragma omp parallel for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            size_t idx_a = off_a_base;
            size_t idx_b = off_b_base;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                size_t dim_sz = out_shape_ptr[d];
                size_t coord = rem % dim_sz;
                rem /= dim_sz;
                if (sha_ptr[d] > 1) idx_a += coord * sa_ptr[d];
                if (shb_ptr[d] > 1) idx_b += coord * sb_ptr[d];
            }
            // FIX: impl->data
            double va = read_scalar_at(a.impl->data->data.get(), idx_a, a._dtype());
            double vb = read_scalar_at(b.impl->data->data.get(), idx_b, b._dtype());
            write_scalar_at(out.impl->data->data.get(), flat, out._dtype(), op(va, vb));
        }
    }
    return out;
}

// --- UNARY OPERATOR KERNEL ---
template <typename Func>
Tensor unary_op_impl(const Tensor& a, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    if (!a.impl) throw std::runtime_error("unary_op: null input");
    
    bool req = a.requires_grad();
    Tensor out(a.shape(), a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;
    
    size_t n = a.numel();
    size_t ndim = a.impl->ndim;
    // FIX: use .data()
    const size_t* shape = a.impl->shape.data();
    const size_t* strides = a.impl->strides.data();
    size_t offset_base = a.impl->offset;
    
    if (a._dtype() == DType::Float32) {
        // FIX: impl->data
        float* data_a = (float*)a.impl->data->data.get();
        float* data_out = (float*)out.impl->data->data.get();
        
        #pragma omp parallel for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            size_t idx = offset_base;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                size_t coord = rem % shape[d];
                rem /= shape[d];
                idx += coord * strides[d];
            }
            data_out[flat] = (float)op((double)data_a[idx]);
        }
    } else {
        #pragma omp parallel for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            size_t idx = offset_base;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                size_t coord = rem % shape[d];
                rem /= shape[d];
                idx += coord * strides[d];
            }
            // FIX: impl->data
            double val = read_scalar_at(a.impl->data->data.get(), idx, a._dtype());
            write_scalar_at(out.impl->data->data.get(), flat, out._dtype(), op(val));
        }
    }
    return out;
}

//                                   IMPLEMENTATIONS

// --- Binary Ops ---

Tensor add_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](double x, double y){ return x + y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradAdd>(a, b) : nullptr);
}

Tensor diff_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](double x, double y){ return x - y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradSub>(a, b) : nullptr);
}

Tensor mult_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](double x, double y){ return x * y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradMul>(a, b) : nullptr);
}

Tensor div_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](double x, double y){ return x / y; }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradDiv>(a, b) : nullptr);
}

Tensor pow_mp(const Tensor& a, const Tensor& b) {
    return binary_op_impl(a, b, [](double x, double y){ return std::pow(x, y); }, 
                          (a.requires_grad() || b.requires_grad()) ? std::make_shared<GradPow>(a, b) : nullptr);
}

// --- Scalar Ops ---

Tensor add_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](double x){ return x + scalar; }, 
                         a.requires_grad() ? std::make_shared<GradAddScalar>(a, scalar) : nullptr);
}

Tensor sub_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](double x){ return x - scalar; }, 
                         a.requires_grad() ? std::make_shared<GradSubScalar>(a, scalar) : nullptr);
}

Tensor sub_afterscalar_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](double x){ return scalar - x; }, 
                         a.requires_grad() ? std::make_shared<GradSubAfterScalar>(a, scalar) : nullptr);
}

Tensor mult_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](double x){ return x * scalar; }, 
                         a.requires_grad() ? std::make_shared<GradMulScalar>(a, scalar) : nullptr);
}

Tensor div_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](double x){ return x / scalar; }, 
                         a.requires_grad() ? std::make_shared<GradDivScalar>(a, scalar) : nullptr);
}

Tensor scalar_div_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](double x){ return scalar / x; }, 
                         a.requires_grad() ? std::make_shared<GradScalarDiv>(a, scalar) : nullptr);
}

Tensor pow_scalar_mp(const Tensor& a, double scalar) {
    return unary_op_impl(a, [scalar](double x){ return std::pow(x, scalar); }, 
                         a.requires_grad() ? std::make_shared<GradPowScalar>(a, scalar) : nullptr);
}

Tensor scalar_pow_mp(double scalar, const Tensor& a) {
    return unary_op_impl(a, [scalar](double x){ return std::pow(scalar, x); }, 
                         a.requires_grad() ? std::make_shared<GradScalarPow>(a, scalar) : nullptr);
}

// --- MatMul ---

Tensor matmul_mp(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim < 2 || B.impl->ndim < 2)
        throw std::runtime_error("matmul_mp requires at least 2D tensors");

    size_t K = A.impl->shape[A.impl->ndim - 1];
    
    // --- DEBUGGING OUTPUT ---
    if (B.impl->shape[B.impl->ndim - 2] != K) {
        throw std::runtime_error("matmul_mp shape mismatch");
    }
    // -------------------------
    
    size_t M = A.impl->shape[A.impl->ndim - 2];
    size_t N = B.impl->shape[B.impl->ndim - 1];
    
    std::vector<size_t> res_shape = A.shape();
    res_shape.back() = N; 
    
    size_t batch_A = 1;
    size_t batch_B = 1;
    for(size_t i=0; i<A.impl->ndim-2; ++i) batch_A *= A.impl->shape[i];
    for(size_t i=0; i<B.impl->ndim-2; ++i) batch_B *= B.impl->shape[i];
    
    size_t batch_out = std::max(batch_A, batch_B);
    
    if (batch_A != batch_B && batch_A != 1 && batch_B != 1) {
        throw std::runtime_error("matmul_mp: batch dimensions not broadcastable");
    }

    bool req = A.requires_grad() || B.requires_grad();
    Tensor C(res_shape, A._dtype(), req);
    
    if (req) C.impl->grad_fn = std::make_shared<GradMatMul>(A, B);
    
    size_t stride_am = A.impl->strides[A.impl->ndim - 2];
    size_t stride_ak = A.impl->strides[A.impl->ndim - 1];
    size_t stride_bk = B.impl->strides[B.impl->ndim - 2];
    size_t stride_bn = B.impl->strides[B.impl->ndim - 1];
    size_t stride_cm = C.impl->strides[C.impl->ndim - 2]; 
    size_t stride_cn = C.impl->strides[C.impl->ndim - 1]; 
    
    // FIX: impl->data
    auto* data_a = A.impl->data->data.get();
    auto* data_b = B.impl->data->data.get();
    auto* data_c = C.impl->data->data.get();
    DType dt = A._dtype();

    size_t stride_A_batch = 0;
    size_t stride_B_batch = 0;
    
    if (A.impl->ndim > 2 && batch_A > 1) stride_A_batch = A.impl->strides[0];
    if (B.impl->ndim > 2 && batch_B > 1) stride_B_batch = B.impl->strides[0];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_out; ++b) {
        for (size_t m = 0; m < M; ++m) {
            size_t c_base = C.impl->offset + b * (M*N); 
            size_t a_batch_off = b * stride_A_batch;
            size_t b_batch_off = b * stride_B_batch;
            
            for (size_t n_idx = 0; n_idx < N; ++n_idx) {
                double sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    size_t idx_a = A.impl->offset + a_batch_off + m * stride_am + k * stride_ak;
                    size_t idx_b = B.impl->offset + b_batch_off + k * stride_bk + n_idx * stride_bn;
                    
                    double va = read_scalar_at(data_a, idx_a, dt);
                    double vb = read_scalar_at(data_b, idx_b, dt);
                    sum += va * vb;
                }
                size_t idx_c = c_base + m * stride_cm + n_idx * stride_cn;
                write_scalar_at(data_c, idx_c, dt, sum);
            }
        }
    }
    return C;
}
// --- Unary Math Ops ---

Tensor abs_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::abs(x); }, a.requires_grad() ? std::make_shared<GradAbs>(a) : nullptr); }
Tensor ln_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::log(x); }, a.requires_grad() ? std::make_shared<GradLn>(a) : nullptr); }
Tensor exp_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::exp(x); }, a.requires_grad() ? std::make_shared<GradExp>(a) : nullptr); }
Tensor sqrt_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::sqrt(x); }, a.requires_grad() ? std::make_shared<GradSqrt>(a) : nullptr); }
Tensor sin_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::sin(x); }, a.requires_grad() ? std::make_shared<GradSin>(a) : nullptr); }
Tensor cos_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::cos(x); }, a.requires_grad() ? std::make_shared<GradCos>(a) : nullptr); }
Tensor tan_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::tan(x); }, a.requires_grad() ? std::make_shared<GradTan>(a) : nullptr); }
Tensor asin_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::asin(x); }, a.requires_grad() ? std::make_shared<GradASin>(a) : nullptr); }
Tensor acos_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::acos(x); }, a.requires_grad() ? std::make_shared<GradACos>(a) : nullptr); }
Tensor atan_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::atan(x); }, a.requires_grad() ? std::make_shared<GradATan>(a) : nullptr); }
Tensor tanh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::tanh(x); }, a.requires_grad() ? std::make_shared<GradTanh>(a) : nullptr); } // Fixed Typo GradTanH -> GradTanh
Tensor sinh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::sinh(x); }, a.requires_grad() ? std::make_shared<GradSinh>(a) : nullptr); } // Fixed Typo GradSinH -> GradSinh
Tensor cosh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::cosh(x); }, a.requires_grad() ? std::make_shared<GradCosh>(a) : nullptr); } // Fixed Typo GradCosH -> GradCosh

Tensor sigmoid_mp(const Tensor& a) { 
    return unary_op_impl(a, [](double x){ return 1.0 / (1.0 + std::exp(-x)); }, 
                         a.requires_grad() ? std::make_shared<GradSigmoid>(a) : nullptr); 
}

Tensor Relu_mp(const Tensor& a) { 
    return unary_op_impl(a, [](double x){ return x > 0 ? x : 0.0; }, 
                         a.requires_grad() ? std::make_shared<GradRelu>(a) : nullptr); 
}

Tensor softplus_mp(const Tensor& a) { 
    return unary_op_impl(a, [](double x){ return std::log(1.0 + std::exp(x)); }, 
                         a.requires_grad() ? std::make_shared<GradSoftplus>(a) : nullptr); // Fixed Typo GradSoftPlus -> GradSoftplus
}

// --- Reductions ---

template<typename ReduceFunc, typename InitFunc>
Tensor reduction_op_impl(const Tensor& t, int dim, ReduceFunc reducer, InitFunc get_init, std::shared_ptr<GradFn> grad_fn) {
    if (!t.impl) throw std::runtime_error("reduction: null input");
    
    int ndim = (int)t.impl->ndim;
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) { /* err */ }

    std::vector<size_t> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) out_shape.push_back(t.impl->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out(out_shape, t._dtype(), t.requires_grad());
    if (t.requires_grad() && grad_fn) {
        out.impl->grad_fn = grad_fn; 
    }

    size_t out_n = out.numel();
    size_t reduce_size = t.impl->shape[dim];
    
    #pragma omp parallel for
    for (size_t i = 0; i < out_n; ++i) {
        size_t rem = i;
        size_t in_base_offset = t.impl->offset;
        
        for (int in_d = ndim - 1; in_d >= 0; --in_d) {
            if (in_d == dim) continue; 
            size_t sz = t.impl->shape[in_d];
            size_t coord = rem % sz;
            rem /= sz;
            in_base_offset += coord * t.impl->strides[in_d];
        }
        
        double acc = get_init();
        size_t stride_reduce = t.impl->strides[dim];
        
        for (size_t k = 0; k < reduce_size; ++k) {
            size_t final_idx = in_base_offset + k * stride_reduce;
            // FIX: impl->data
            double val = read_scalar_at(t.impl->data->data.get(), final_idx, t._dtype());
            acc = reducer(acc, val);
        }
        // FIX: impl->data
        write_scalar_at(out.impl->data->data.get(), i, out._dtype(), acc);
    }
    return out;
}

Tensor sum_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, 
        [](double acc, double val){ return acc + val; }, 
        []{ return 0.0; }, 
        t.requires_grad() ? std::make_shared<GradSum>(t, dim) : nullptr);
}

Tensor max_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, 
        [](double acc, double val){ return std::max(acc, val); }, 
        []{ return -std::numeric_limits<double>::infinity(); }, 
        nullptr); // No GradMax yet
}

Tensor min_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, 
        [](double acc, double val){ return std::min(acc, val); }, 
        []{ return std::numeric_limits<double>::infinity(); }, 
        nullptr); // No GradMin yet
}

Tensor mean_mp(const Tensor& t, int dim) {
    Tensor s = sum_mp(t, dim);
    double count = (double)t.impl->shape[dim < 0 ? dim + t.impl->ndim : dim];
    return mult_scalar_mp(s, 1.0 / count);
}
// --- Comparisons ---

Tensor lt_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x < b ? 1.0 : 0.0; }); }
Tensor le_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x <= b ? 1.0 : 0.0; }); }
Tensor gt_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x > b ? 1.0 : 0.0; }); }
Tensor ge_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x >= b ? 1.0 : 0.0; }); }
Tensor eq_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x == b ? 1.0 : 0.0; }); }
Tensor neq_mp(const Tensor& a, double b) { return unary_op_impl(a, [b](double x){ return x != b ? 1.0 : 0.0; }); }

Tensor lt_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x < y ? 1.0 : 0.0; }); }
Tensor le_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x <= y ? 1.0 : 0.0; }); }
Tensor gt_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x > y ? 1.0 : 0.0; }); }
Tensor ge_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x >= y ? 1.0 : 0.0; }); }
Tensor eq_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x == y ? 1.0 : 0.0; }); }
Tensor ne_mp(const Tensor& a, const Tensor& b) { return binary_op_impl(a, b, [](double x, double y){ return x != y ? 1.0 : 0.0; }); }

// --- Utilities ---

Tensor cat_mp(const std::vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) throw std::runtime_error("cat_mp: empty list");
    
    std::vector<size_t> out_shape = tensors[0].shape();
    size_t dim_sum = 0;
    for (const auto& t : tensors) {
        if (t.impl->ndim != out_shape.size()) throw std::runtime_error("cat_mp: dim mismatch");
        dim_sum += t.impl->shape[dim];
    }
    out_shape[dim] = dim_sum;
    
    Tensor out(out_shape, tensors[0]._dtype(), false); 
    
    size_t current_offset_dim = 0;
    for (const auto& t : tensors) {
        // Placeholder for cat logic
        // Implementing parallel copy for cat is similar to clone but with offset logic
        // Skipping full implementation for brevity as requested
        current_offset_dim += t.impl->shape[dim];
    }
    return out;
}

// --- Compound assignment operators ---

Tensor& operator+=(Tensor& a, const Tensor& b) {
    a = add_mp(a, b);
    return a;
}
Tensor& operator+=(Tensor& a, double scalar) {
    a = add_scalar_mp(a, scalar);
    return a;
}

Tensor& operator-=(Tensor& a, const Tensor& b) {
    a = diff_mp(a, b);
    return a;
}
Tensor& operator-=(Tensor& a, double scalar) {
    a = sub_scalar_mp(a, scalar);
    return a;
}

Tensor& operator*=(Tensor& a, const Tensor& b) {
    a = mult_mp(a, b);
    return a;
}
Tensor& operator*=(Tensor& a, double scalar) {
    a = mult_scalar_mp(a, scalar);
    return a;
}

Tensor& operator/=(Tensor& a, const Tensor& b) {
    a = div_mp(a, b);
    return a;
}
Tensor& operator/=(Tensor& a, double scalar) {
    a = div_scalar_mp(a, scalar);
    return a;
}

Tensor& operator^=(Tensor& a, const Tensor& b) {
    a = pow_mp(a, b);
    return a;
}
Tensor& operator^=(Tensor& a, double scalar) {
    a = pow_scalar_mp(a, scalar);
    return a;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return add_mp(a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return diff_mp(a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return mult_mp(a, b);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return div_mp(a, b);
}

Tensor operator^(const Tensor& a, const Tensor& b) {
    return pow_mp(a, b);
}

// scalar on the left
Tensor operator+(double s, const Tensor& a) {
    return add_scalar_mp(a, s);
}
Tensor operator+(const Tensor& a,double s) {
    return add_scalar_mp(a, s);
}