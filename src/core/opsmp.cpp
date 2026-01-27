// Standard headers FIRST to prevent issues if local headers are malformed
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstring> 

// Local headers AFTER
#include "opsmp.h"
#include "autograd.h"

// ======================================================================================
//                                      HELPERS
// ======================================================================================

// Simple SVO-friendly broadcast check
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
//                            OPTIMIZED ITERATOR (NO MODULO)
// ======================================================================================

struct TensorIterator {
    // We flatten dimensions to reduce loop overhead
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides_a; // 0 if broadcasted
    std::vector<size_t> strides_b; // 0 if broadcasted
    size_t numel;

    TensorIterator(const Tensor& a, const Tensor& b, const std::vector<size_t>& out_shape) {
        size_t out_ndim = out_shape.size();
        numel = 1;
        for(auto s : out_shape) numel *= s;

        // Pad shapes to match output ndim
        std::vector<size_t> shape_a_pad(out_ndim, 1);
        std::vector<size_t> strides_a_pad(out_ndim, 0);
        size_t off_a = out_ndim - a.impl->ndim;
        for(size_t i=0; i<a.impl->ndim; ++i) {
            shape_a_pad[off_a + i] = a.impl->shape[i];
            strides_a_pad[off_a + i] = a.impl->strides[i];
        }

        std::vector<size_t> shape_b_pad(out_ndim, 1);
        std::vector<size_t> strides_b_pad(out_ndim, 0);
        size_t off_b = out_ndim - b.impl->ndim;
        for(size_t i=0; i<b.impl->ndim; ++i) {
            shape_b_pad[off_b + i] = b.impl->shape[i];
            strides_b_pad[off_b + i] = b.impl->strides[i];
        }

        // DIMENSION FOLDING: Coalesce contiguous dimensions
        // We start from the back (innermost dim)
        if (out_ndim > 0) {
            shape.push_back(out_shape.back());
            // Broadcast stride logic: if dim is 1, stride is effectively 0 for movement logic 
            // BUT for folding we need to be careful.
            // Simplified broadcast stride: if dim size is 1, set stride to 0.
            size_t sa = (shape_a_pad.back() == 1) ? 0 : strides_a_pad.back();
            size_t sb = (shape_b_pad.back() == 1) ? 0 : strides_b_pad.back();
            
            strides_a.push_back(sa);
            strides_b.push_back(sb);

            for (int i = (int)out_ndim - 2; i >= 0; --i) {
                size_t dim_sz = out_shape[i];
                size_t sa_curr = (shape_a_pad[i] == 1) ? 0 : strides_a_pad[i];
                size_t sb_curr = (shape_b_pad[i] == 1) ? 0 : strides_b_pad[i];

                // Check if we can fold:
                // 1. Not a broadcast boundary (dim_sz match) or handled by stride 0
                // 2. Contiguous in memory: stride_curr == stride_prev * size_prev
                
                size_t prev_sz = shape.back();
                size_t sa_prev = strides_a.back();
                size_t sb_prev = strides_b.back();

                // Fold Condition:
                // stride_curr == stride_prev * prev_sz OR (stride_curr == 0 && sa_prev == 0)
                // This is slightly complex with broadcasting. 
                // Simple heuristic: Only fold if strictly contiguous or both broadcasted.
                
                bool fold_a = (sa_curr == sa_prev * prev_sz);
                if (sa_curr == 0 && sa_prev == 0) fold_a = true;
                
                bool fold_b = (sb_curr == sb_prev * prev_sz);
                if (sb_curr == 0 && sb_prev == 0) fold_b = true;

                if (fold_a && fold_b) {
                    shape.back() *= dim_sz;
                    // Strides remain the stride of the inner-most part of the block
                } else {
                    shape.push_back(dim_sz);
                    strides_a.push_back(sa_curr);
                    strides_b.push_back(sb_curr);
                }
            }
            // Reverse back to [outer, ..., inner]
            std::reverse(shape.begin(), shape.end());
            std::reverse(strides_a.begin(), strides_a.end());
            std::reverse(strides_b.begin(), strides_b.end());
        }
        ndim = shape.size();
    }
};

// ======================================================================================
//                               TEMPLATE KERNELS
// ======================================================================================

// --- BINARY OPERATOR KERNEL ---
template <typename Func>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, Func op, std::shared_ptr<GradFn> grad_fn = nullptr) {
    if (!a.impl || !b.impl) throw std::runtime_error("binary_op: null input");
    
    // 1. Shape & Broadcast Analysis
    std::vector<size_t> shape_a = a.shape(); // Access via helper if needed
    std::vector<size_t> shape_b = b.shape();
    
    if (!broadcastable(shape_a, shape_b)) throw std::runtime_error("Shape mismatch in binary op");
    std::vector<size_t> out_shape = broadcast_shape(shape_a, shape_b);
    
    bool req = a.requires_grad() || b.requires_grad();
    Tensor out(out_shape, a._dtype(), req);
    if (req && grad_fn) out.impl->grad_fn = grad_fn;

    // 2. Build Optimized Iterator
    TensorIterator iter(a, b, out_shape);
    size_t n = iter.numel;
    
    const size_t* shape_ptr = iter.shape.data();
    const size_t* stra_ptr = iter.strides_a.data();
    const size_t* strb_ptr = iter.strides_b.data();
    size_t ndim = iter.ndim;
    
    size_t off_a_base = a.impl->offset;
    size_t off_b_base = b.impl->offset;

    // Dispatch based on Type
    if (a._dtype() == DType::Float32) {
        float* ptr_a = (float*)a.impl->data->data.get();
        float* ptr_b = (float*)b.impl->data->data.get();
        float* ptr_out = (float*)out.impl->data->data.get();

        // NUMA-AWARE PARALLELISM
        // Use static schedule to align threads to memory pages
        #pragma omp parallel
        {
            size_t num_threads = omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            size_t chunk = (n + num_threads - 1) / num_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);
            
            if (start < end) {
                // Initial coordinate calculation (Expensive Div/Mod done ONCE per thread)
                std::vector<size_t> coords(ndim, 0);
                size_t temp = start;
                size_t idx_a = off_a_base;
                size_t idx_b = off_b_base;

                for (int d = (int)ndim - 1; d >= 0; --d) {
                    size_t sz = shape_ptr[d];
                    size_t c = temp % sz;
                    temp /= sz;
                    coords[d] = c;
                    idx_a += c * stra_ptr[d];
                    idx_b += c * strb_ptr[d];
                }

                // HOT LOOP: STRIDED INCREMENT
                for (size_t i = start; i < end; ++i) {
                    // Compute
                    ptr_out[i] = (float)op((double)ptr_a[idx_a], (double)ptr_b[idx_b]);

                    // Advance Pointers
                    for (int d = (int)ndim - 1; d >= 0; --d) {
                        coords[d]++;
                        idx_a += stra_ptr[d];
                        idx_b += strb_ptr[d];
                        
                        if (coords[d] < shape_ptr[d]) {
                            break; // No overflow, continue
                        }
                        
                        // Overflow: Reset and carry over
                        coords[d] = 0;
                        idx_a -= shape_ptr[d] * stra_ptr[d];
                        idx_b -= shape_ptr[d] * strb_ptr[d];
                    }
                }
            }
        }
    } else {
        // Fallback for other types (Double etc)
        auto* d_a = a.impl->data->data.get();
        auto* d_b = b.impl->data->data.get();
        auto* d_out = out.impl->data->data.get();
        DType dt = a._dtype();
        DType out_dt = out._dtype();

        #pragma omp parallel
        {
            size_t num_threads = omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            size_t chunk = (n + num_threads - 1) / num_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);

            if (start < end) {
                std::vector<size_t> coords(ndim, 0);
                size_t temp = start;
                size_t idx_a = off_a_base;
                size_t idx_b = off_b_base;

                for (int d = (int)ndim - 1; d >= 0; --d) {
                    size_t sz = shape_ptr[d];
                    size_t c = temp % sz;
                    temp /= sz;
                    coords[d] = c;
                    idx_a += c * stra_ptr[d];
                    idx_b += c * strb_ptr[d];
                }

                for (size_t i = start; i < end; ++i) {
                    double va = read_scalar_at(d_a, idx_a, dt);
                    double vb = read_scalar_at(d_b, idx_b, dt);
                    write_scalar_at(d_out, i, out_dt, op(va, vb));

                    for (int d = (int)ndim - 1; d >= 0; --d) {
                        coords[d]++;
                        idx_a += stra_ptr[d];
                        idx_b += strb_ptr[d];
                        if (coords[d] < shape_ptr[d]) break;
                        coords[d] = 0;
                        idx_a -= shape_ptr[d] * stra_ptr[d];
                        idx_b -= shape_ptr[d] * strb_ptr[d];
                    }
                }
            }
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
    const size_t* shape = a.impl->shape.data();
    const size_t* strides = a.impl->strides.data();
    size_t offset_base = a.impl->offset;
    
    // Fast Path: Contiguous
    // If input is contiguous, we treat it as 1D array -> 1D loop -> perfect vectorization
    if (a.is_contiguous()) {
        if (a._dtype() == DType::Float32) {
            float* data_a = (float*)a.impl->data->data.get();
            float* data_out = (float*)out.impl->data->data.get();
            float* start_ptr = data_a + offset_base;
            
            // OpenMP handles 1D loops very well with static schedule (Chunking)
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                data_out[i] = (float)op((double)start_ptr[i]);
            }
        } else {
            // Generic Contiguous
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                double val = read_scalar_at(a.impl->data->data.get(), offset_base + i, a._dtype());
                write_scalar_at(out.impl->data->data.get(), i, out._dtype(), op(val));
            }
        }
        return out;
    }

    // Slow Path: Non-Contiguous (Strided Iterator)
    // We use the same strided logic as binary op
    if (a._dtype() == DType::Float32) {
        float* data_a = (float*)a.impl->data->data.get();
        float* data_out = (float*)out.impl->data->data.get();
        
        #pragma omp parallel
        {
            size_t num_threads = omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            size_t chunk = (n + num_threads - 1) / num_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);
            
            if (start < end) {
                std::vector<size_t> coords(ndim);
                size_t temp = start;
                size_t idx = offset_base;
                for (int d = (int)ndim - 1; d >= 0; --d) {
                    size_t sz = shape[d];
                    size_t c = temp % sz;
                    temp /= sz;
                    coords[d] = c;
                    idx += c * strides[d];
                }
                
                for (size_t i = start; i < end; ++i) {
                    data_out[i] = (float)op((double)data_a[idx]);
                    
                    for (int d = (int)ndim - 1; d >= 0; --d) {
                        coords[d]++;
                        idx += strides[d];
                        if (coords[d] < shape[d]) break;
                        coords[d] = 0;
                        idx -= shape[d] * strides[d];
                    }
                }
            }
        }
    } else {
        // Generic Non-Contiguous
        #pragma omp parallel
        {
            size_t num_threads = omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            size_t chunk = (n + num_threads - 1) / num_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);
            
            if (start < end) {
                std::vector<size_t> coords(ndim);
                size_t temp = start;
                size_t idx = offset_base;
                for (int d = (int)ndim - 1; d >= 0; --d) {
                    coords[d] = (temp % shape[d]);
                    temp /= shape[d];
                    idx += coords[d] * strides[d];
                }
                
                for (size_t i = start; i < end; ++i) {
                     double val = read_scalar_at(a.impl->data->data.get(), idx, a._dtype());
                     write_scalar_at(out.impl->data->data.get(), i, out._dtype(), op(val));
                     
                     for (int d = (int)ndim - 1; d >= 0; --d) {
                        coords[d]++;
                        idx += strides[d];
                        if (coords[d] < shape[d]) break;
                        coords[d] = 0;
                        idx -= shape[d] * strides[d];
                    }
                }
            }
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

// --- MatMul (Block-Optimized) ---

Tensor matmul_mp(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim < 2 || B.impl->ndim < 2)
        throw std::runtime_error("matmul_mp requires at least 2D tensors");

    size_t K = A.impl->shape[A.impl->ndim - 1];
    if (B.impl->shape[B.impl->ndim - 2] != K) throw std::runtime_error("matmul_mp shape mismatch");
    
    size_t M = A.impl->shape[A.impl->ndim - 2];
    size_t N = B.impl->shape[B.impl->ndim - 1];
    
    std::vector<size_t> res_shape = A.shape();
    res_shape.back() = N; 
    
    size_t batch_A = 1; for(size_t i=0; i<A.impl->ndim-2; ++i) batch_A *= A.impl->shape[i];
    size_t batch_B = 1; for(size_t i=0; i<B.impl->ndim-2; ++i) batch_B *= B.impl->shape[i];
    size_t batch_out = std::max(batch_A, batch_B);
    
    bool req = A.requires_grad() || B.requires_grad();
    Tensor C(res_shape, A._dtype(), req);
    if (req) C.impl->grad_fn = std::make_shared<GradMatMul>(A, B);
    
    // Strides
    size_t stride_am = A.impl->strides[A.impl->ndim - 2];
    size_t stride_ak = A.impl->strides[A.impl->ndim - 1];
    size_t stride_bk = B.impl->strides[B.impl->ndim - 2];
    size_t stride_bn = B.impl->strides[B.impl->ndim - 1];
    size_t stride_cm = C.impl->strides[C.impl->ndim - 2]; 
    size_t stride_cn = C.impl->strides[C.impl->ndim - 1]; 

    // Assuming Float32 for optimization demo, generic fallback exists
    if (A._dtype() == DType::Float32) {
        float* data_a = (float*)A.impl->data->data.get();
        float* data_b = (float*)B.impl->data->data.get();
        float* data_c = (float*)C.impl->data->data.get();

        size_t stride_A_batch = (A.impl->ndim > 2 && batch_A > 1) ? A.impl->strides[0] : 0;
        size_t stride_B_batch = (B.impl->ndim > 2 && batch_B > 1) ? B.impl->strides[0] : 0;

        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < batch_out; ++b) {
            size_t c_base = C.impl->offset + b * (M*N); 
            size_t a_batch_off = b * stride_A_batch; // Simplified batching (assumes contig batch dim)
            size_t b_batch_off = b * stride_B_batch;
            
            // Simple Cache Blocking (Tiling)
            // Block sizes fitting L1/L2
            const size_t BLOCK_M = 64;
            const size_t BLOCK_N = 64; 
            const size_t BLOCK_K = 64;

            for (size_t m0 = 0; m0 < M; m0 += BLOCK_M) {
                size_t m_end = std::min(m0 + BLOCK_M, M);
                for (size_t n0 = 0; n0 < N; n0 += BLOCK_N) {
                    size_t n_end = std::min(n0 + BLOCK_N, N);
                    
                    for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
                        size_t k_end = std::min(k0 + BLOCK_K, K);
                        
                        // Inner Loops
                        for (size_t m = m0; m < m_end; ++m) {
                            for (size_t n_idx = n0; n_idx < n_end; ++n_idx) {
                                float sum = 0.0f;
                                if (k0 == 0) sum = 0.0f; 
                                else sum = data_c[c_base + m * stride_cm + n_idx * stride_cn];

                                for (size_t k = k0; k < k_end; ++k) {
                                    size_t idx_a = A.impl->offset + a_batch_off + m * stride_am + k * stride_ak;
                                    size_t idx_b = B.impl->offset + b_batch_off + k * stride_bk + n_idx * stride_bn;
                                    sum += data_a[idx_a] * data_b[idx_b];
                                }
                                data_c[c_base + m * stride_cm + n_idx * stride_cn] = sum;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Fallback generic
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_out; ++b) {
            for (size_t m = 0; m < M; ++m) {
                size_t c_base = C.impl->offset + b * (M*N); 
                size_t a_batch_off = (batch_A>1) ? b * A.impl->strides[0] : 0;
                size_t b_batch_off = (batch_B>1) ? b * B.impl->strides[0] : 0;
                for (size_t n_idx = 0; n_idx < N; ++n_idx) {
                    double sum = 0;
                    for (size_t k = 0; k < K; ++k) {
                        size_t idx_a = A.impl->offset + a_batch_off + m * stride_am + k * stride_ak;
                        size_t idx_b = B.impl->offset + b_batch_off + k * stride_bk + n_idx * stride_bn;
                        sum += read_scalar_at(A.impl->data->data.get(), idx_a, A._dtype()) * read_scalar_at(B.impl->data->data.get(), idx_b, B._dtype());
                    }
                    write_scalar_at(C.impl->data->data.get(), c_base + m * stride_cm + n_idx * stride_cn, C._dtype(), sum);
                }
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
Tensor tanh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::tanh(x); }, a.requires_grad() ? std::make_shared<GradTanh>(a) : nullptr); }
Tensor sinh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::sinh(x); }, a.requires_grad() ? std::make_shared<GradSinh>(a) : nullptr); }
Tensor cosh_mp(const Tensor& a) { return unary_op_impl(a, [](double x){ return std::cosh(x); }, a.requires_grad() ? std::make_shared<GradCosh>(a) : nullptr); }

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
                         a.requires_grad() ? std::make_shared<GradSoftplus>(a) : nullptr); 
}

// --- Reductions ---

template<typename ReduceFunc, typename InitFunc>
Tensor reduction_op_impl(const Tensor& t, int dim, ReduceFunc reducer, InitFunc get_init, std::shared_ptr<GradFn> grad_fn) {
    if (!t.impl) throw std::runtime_error("reduction: null input");
    
    int ndim = (int)t.impl->ndim;
    if (dim < 0) dim += ndim;
    
    // Output shape logic
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
    
    size_t stride_reduce = t.impl->strides[dim];
    size_t offset_base = t.impl->offset;
    
    const size_t* t_strides = t.impl->strides.data();
    const size_t* out_shape_ptr = out.impl->shape.data(); 
    std::vector<size_t> eff_strides;
    for(int d=0; d<ndim; ++d) { if(d!=dim) eff_strides.push_back(t_strides[d]); }
    
    const size_t* eff_strides_ptr = eff_strides.data();
    size_t eff_ndim = eff_strides.size();

    #pragma omp parallel
    {
        size_t num_threads = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t chunk = (out_n + num_threads - 1) / num_threads;
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, out_n);
        
        if (start < end) {
             std::vector<size_t> coords(eff_ndim, 0);
             size_t temp = start;
             size_t current_base = offset_base;
             
             // Init Coords
             for(int d = (int)eff_ndim - 1; d >= 0; --d) {
                 size_t sz = (out_shape_ptr[d]); 
                 size_t c = temp % sz;
                 temp /= sz;
                 coords[d] = c;
                 current_base += c * eff_strides_ptr[d];
             }

             for (size_t i = start; i < end; ++i) {
                 double acc = get_init();
                 
                 for (size_t k = 0; k < reduce_size; ++k) {
                     size_t final_idx = current_base + k * stride_reduce;
                     double val = read_scalar_at(t.impl->data->data.get(), final_idx, t._dtype());
                     acc = reducer(acc, val);
                 }
                 write_scalar_at(out.impl->data->data.get(), i, out._dtype(), acc);

                 // Advance Base
                 for (int d = (int)eff_ndim - 1; d >= 0; --d) {
                     coords[d]++;
                     current_base += eff_strides_ptr[d];
                     if (coords[d] < out_shape_ptr[d]) break;
                     coords[d] = 0;
                     current_base -= out_shape_ptr[d] * eff_strides_ptr[d];
                 }
             }
        }
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
        nullptr);
}

Tensor min_mp(const Tensor& t, int dim) {
    return reduction_op_impl(t, dim, 
        [](double acc, double val){ return std::min(acc, val); }, 
        []{ return std::numeric_limits<double>::infinity(); }, 
        nullptr);
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
    size_t ndim = out_shape.size();
    if (dim >= ndim) throw std::out_of_range("cat_mp: dim out of range");

    size_t dim_sum = 0;
    for (const auto& t : tensors) {
        if (t.impl->ndim != ndim) throw std::runtime_error("cat_mp: ndim mismatch");
        for (size_t i = 0; i < ndim; ++i) {
            if (i != dim && t.impl->shape[i] != out_shape[i]) 
                throw std::runtime_error("cat_mp: shape mismatch outside cat dim");
        }
        dim_sum += t.impl->shape[dim];
    }
    out_shape[dim] = dim_sum;

    Tensor out(out_shape, tensors[0]._dtype(), false);
    char* out_data_ptr = (char*)out.impl->data->data.get();
    size_t dtype_sz = out.dtype_bytes();

    size_t outer_elements = 1;
    for (size_t i = 0; i < dim; ++i) outer_elements *= out_shape[i];
    
    // Fast path: All Contiguous?
    bool all_contiguous = true;
    for(auto& t : tensors) if(!t.is_contiguous()) all_contiguous = false;

    if (all_contiguous) {
        // Block Copy Logic for contiguous chunks
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < outer_elements; ++o) {
            size_t block_size_bytes = (out.numel() / outer_elements) * dtype_sz;
            char* out_ptr_base = out_data_ptr + o * block_size_bytes;
            
            size_t current_offset_bytes = 0;
            for (const auto& t : tensors) {
                size_t t_bytes = (t.numel() / outer_elements) * dtype_sz;
                const char* t_data = (const char*)t.impl->data->data.get() + t.impl->offset * dtype_sz + o * t_bytes;
                
                std::memcpy(out_ptr_base + current_offset_bytes, t_data, t_bytes);
                current_offset_bytes += t_bytes;
            }
        }
    } else {
        // Slow path: Non-contiguous element-wise copy
        std::vector<size_t> dim_offsets(tensors.size(), 0);
        size_t current = 0;
        for(size_t i=0; i<tensors.size(); ++i) {
            dim_offsets[i] = current;
            current += tensors[i].impl->shape[dim];
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tensors.size(); ++i) {
            const Tensor& t = tensors[i];
            size_t dim_off = dim_offsets[i];
            size_t n = t.numel();
            const size_t* t_shape = t.impl->shape.data();
            const size_t* t_strides = t.impl->strides.data();
            size_t t_ndim = t.impl->ndim;
            size_t t_offset_base = t.impl->offset;

            // Manual loop over t elements with coordinate mapping
            for(size_t j=0; j<n; ++j) {
                size_t temp = j;
                size_t t_idx = t_offset_base;
                size_t out_idx = 0;
                
                for(int d = (int)t_ndim - 1; d >= 0; --d) {
                    size_t sz = t_shape[d];
                    size_t coord = temp % sz;
                    temp /= sz;
                    t_idx += coord * t_strides[d];
                    
                    size_t out_coord = coord;
                    if (d == (int)dim) out_coord += dim_off;
                    out_idx += out_coord * out.impl->strides[d];
                }
                
                double val = read_scalar_at(t.impl->data->data.get(), t_idx, t._dtype());
                write_scalar_at(out.impl->data->data.get(), out_idx, out._dtype(), val);
            }
        }
    }
    return out;
}