#include "opsmp.h"
#include "autograd.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>

// ======================================================================================
//                                      HELPERS
// ======================================================================================
static const size_t MP_OMP_THRESHOLD = 8192; 

// Odometer for Generic Types
static const int MAX_ITER_DIMS = 32;

struct GenericOdometer {
    int ndim;
    const size_t* shape;
    size_t strides_a[MAX_ITER_DIMS];
    size_t strides_b[MAX_ITER_DIMS];
    size_t coords[MAX_ITER_DIMS];
    size_t offset_a;
    size_t offset_b;

    GenericOdometer(int nd, const size_t* sh, const std::vector<size_t>& sa, const std::vector<size_t>& sb)
        : ndim(nd), shape(sh), offset_a(0), offset_b(0) {
        
        if (nd > MAX_ITER_DIMS) throw std::runtime_error("Rank too high for stack");

        for(int i=0; i<nd; ++i) {
            strides_a[i] = sa[i];
            strides_b[i] = sb[i];
            coords[i] = 0;
        }
    }

    void init(size_t linear_idx) {
        offset_a = 0; offset_b = 0;
        size_t rem = linear_idx;
        for (int i = ndim - 1; i >= 0; --i) {
            size_t c = rem % shape[i];
            rem /= shape[i];
            coords[i] = c;
            offset_a += c * strides_a[i];
            offset_b += c * strides_b[i];
        }
    }

    void next() {
        for (int i = ndim - 1; i >= 0; --i) {
            coords[i]++;
            offset_a += strides_a[i];
            offset_b += strides_b[i];
            if (coords[i] < shape[i]) return;
            
            coords[i] = 0;
            offset_a -= strides_a[i] * shape[i]; // Reset
            offset_b -= strides_b[i] * shape[i];
        }
    }
};

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

template <typename Func>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    std::vector<size_t> shape_a = a.shape();
    std::vector<size_t> shape_b = b.shape();
    std::vector<size_t> out_shape = broadcast_shape(shape_a, shape_b);
    
    bool req = a.requires_grad() || b.requires_grad();
    Tensor out(out_shape, a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;

    size_t n = out.numel();
    int ndim = (int)out_shape.size();

    std::vector<size_t> sa(ndim, 0), sb(ndim, 0);
    int off_a = ndim - (int)shape_a.size();
    int off_b = ndim - (int)shape_b.size();
    
    for(int i=0; i< (int)shape_a.size(); ++i) if(shape_a[i] > 1) sa[i+off_a] = a.impl->strides[i];
    for(int i=0; i< (int)shape_b.size(); ++i) if(shape_b[i] > 1) sb[i+off_b] = b.impl->strides[i];

    if (a._dtype() == DType::Float32) {
        float* pa = (float*)a.impl->data->data.get();
        float* pb = (float*)b.impl->data->data.get();
        float* po = (float*)out.impl->data->data.get();
        size_t base_a = a.impl->offset;
        size_t base_b = b.impl->offset;

        if (n < MP_OMP_THRESHOLD) {
            GenericOdometer it(ndim, out_shape.data(), sa, sb);
            for(size_t i=0; i<n; ++i) {
                po[i] = (float)op((double)pa[base_a + it.offset_a], (double)pb[base_b + it.offset_b]);
                it.next();
            }
        } else {
            #pragma omp parallel
            {
                GenericOdometer it(ndim, out_shape.data(), sa, sb);
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                size_t chunk = (n + nth - 1) / nth;
                size_t start = std::min(chunk*tid, n);
                size_t end = std::min(chunk*(tid+1), n);
                
                if (start < end) {
                    it.init(start);
                    for(size_t i=start; i<end; ++i) {
                        po[i] = (float)op((double)pa[base_a + it.offset_a], (double)pb[base_b + it.offset_b]);
                        it.next();
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for if(n > MP_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            size_t temp = i;
            size_t idx_a = a.impl->offset;
            size_t idx_b = b.impl->offset;
            for(int d=ndim-1; d>=0; --d) {
                size_t sz = out_shape[d];
                size_t coord = temp % sz;
                temp /= sz;
                if (d >= off_a && shape_a[d-off_a] > 1) idx_a += coord * a.impl->strides[d-off_a];
                if (d >= off_b && shape_b[d-off_b] > 1) idx_b += coord * b.impl->strides[d-off_b];
            }
            double val_a = read_scalar_at(a.impl->data->data.get(), idx_a, a._dtype());
            double val_b = read_scalar_at(b.impl->data->data.get(), idx_b, b._dtype());
            write_scalar_at(out.impl->data->data.get(), i, out._dtype(), op(val_a, val_b));
        }
    }
    return out;
}

// --- UNARY ---
template <typename Func>
Tensor unary_op_impl(const Tensor& a, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    bool req = a.requires_grad();
    Tensor out(a.shape(), a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;
    
    size_t n = a.numel();
    
    // Contiguous fast path
    if (a.is_contiguous()) {
        if (a._dtype() == DType::Float32) {
            float* pa = (float*)a.impl->data->data.get() + a.impl->offset;
            float* po = (float*)out.impl->data->data.get();
            #pragma omp parallel for if(n > MP_OMP_THRESHOLD)
            for(size_t i=0; i<n; ++i) po[i] = (float)op((double)pa[i]);
        } else {
             #pragma omp parallel for if(n > MP_OMP_THRESHOLD)
             for(size_t i=0; i<n; ++i) {
                 double v = read_scalar_at(a.impl->data->data.get(), a.impl->offset + i, a._dtype());
                 write_scalar_at(out.impl->data->data.get(), i, out._dtype(), op(v));
             }
        }
    } else {
        // Non-contiguous: use generic iterator or just clone then op
        Tensor ac = a.contiguous();
        return unary_op_impl(ac, op, grad_fn);
    }
    return out;
}

// ---------------------- Implementations ----------------------

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

// --- Matmul (Scalar/MP Fallback) ---
Tensor matmul_mp(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim < 2 || B.impl->ndim < 2) throw std::runtime_error("matmul_mp: needs 2D");
    
    size_t M = A.impl->shape[A.impl->ndim - 2];
    size_t K = A.impl->shape[A.impl->ndim - 1];
    size_t N = B.impl->shape[B.impl->ndim - 1];
    if (B.impl->shape[B.impl->ndim - 2] != K) throw std::runtime_error("matmul_mp shape mismatch");
    
    std::vector<size_t> res_shape = A.shape();
    res_shape.back() = N;
    
    Tensor C(res_shape, A._dtype(), A.requires_grad() || B.requires_grad());
    if (C.requires_grad()) C.impl->grad_fn = std::make_shared<GradMatMul>(A, B);

    // Naive triple loop for fallback
    float* c_ptr = (float*)C.impl->data->data.get();
    const float* a_ptr = (const float*)A.impl->data->data.get();
    const float* b_ptr = (const float*)B.impl->data->data.get();
    
    #pragma omp parallel for collapse(2) if(M*N*K > MP_OMP_THRESHOLD)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += a_ptr[i*K + k] * b_ptr[k*N + j];
            }
            c_ptr[i*N + j] = sum;
        }
    }
    return C;
}

// --- Unary Math ---
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
Tensor tanh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::tanh(x); }, a.requires_grad() ? std::make_shared<GradTanh>(a) : nullptr); }
Tensor sinh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::sinh(x); }, a.requires_grad() ? std::make_shared<GradSinh>(a) : nullptr); }
Tensor cosh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::cosh(x); }, a.requires_grad() ? std::make_shared<GradCosh>(a) : nullptr); }
Tensor sigmoid_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return 1.0 / (1.0 + std::exp(-x)); }, a.requires_grad() ? std::make_shared<GradSigmoid>(a) : nullptr); }
Tensor Relu_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return x > 0 ? x : 0.0; }, a.requires_grad() ? std::make_shared<GradRelu>(a) : nullptr); }
Tensor softplus_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::log(1.0 + std::exp(x)); }, a.requires_grad() ? std::make_shared<GradSoftplus>(a) : nullptr); }

// --- Reductions ---
Tensor sum_mp(const Tensor& t, int dim) {
    size_t n = t.numel();
    double total = 0;
    if (dim == -1) {
         if (t._dtype() == DType::Float32) {
             const float* p = (const float*)t.impl->data->data.get();
             #pragma omp parallel for reduction(+:total)
             for(size_t i=0; i<n; ++i) total += p[i];
         } else {
             #pragma omp parallel for reduction(+:total)
             for(size_t i=0; i<n; ++i) total += read_scalar_at(t.impl->data->data.get(), i, t._dtype());
         }
         Tensor out({1}, t._dtype(), t.requires_grad());
         write_scalar_at(out.impl->data->data.get(), 0, t._dtype(), total);
         if (t.requires_grad()) out.impl->grad_fn = std::make_shared<GradSum>(t, dim);
         return out;
    }
    throw std::runtime_error("sum_mp: dim != -1 not implemented in fallback");
}

Tensor mean_mp(const Tensor& t, int dim) {
    Tensor s = sum_mp(t, dim);
    return mult_scalar_mp(s, 1.0 / t.numel());
}
Tensor max_mp(const Tensor& t, int dim) { return Tensor({1}); } 
Tensor min_mp(const Tensor& t, int dim) { return Tensor({1}); } 

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

Tensor cat_mp(const std::vector<Tensor>& tensors, size_t dim) { return Tensor(); }