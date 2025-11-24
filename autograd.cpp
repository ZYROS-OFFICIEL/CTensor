// autograd_core.h / autograd_core.cpp
#include <unordered_map>
#include <functional>
#include <stack>
#include <set>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "autograd.h"
#include "tensor1.h"
#include "opsmp.h"

// -------------------- helpers --------------------

// ensure grad buffer exists on tensor; if zero=true fill with zeros
// Replace your current ensure_grad_buffer implementation with this.

inline void ensure_grad_buffer(Tensor &t, bool zero_existing) {
    if (!t.impl) throw std::runtime_error("ensure_grad_buffer: tensor undefined");
    if (!t.impl->storage) throw std::runtime_error("ensure_grad_buffer: tensor has no storage");

    // Allocate based on underlying storage->size (stride-safe)
    size_t total_elems = t.impl->storage->size;
    size_t nbytes = total_elems * t.dtype_bytes();

    if (!t.impl->storage->grad) {
        // ALWAYS zero newly allocated buffer (prevents reading uninitialized memory)
        void* ptr = nullptr;
        if (nbytes) {
            ptr = std::malloc(nbytes);
            if (!ptr) throw std::bad_alloc();
            std::memset(ptr, 0, nbytes);   // zero on allocation
        }
        t.impl->storage->grad = std::shared_ptr<void>(ptr, std::free);
    } else if (zero_existing) {
        // Caller explicitly requested zeroing existing buffer
        if (nbytes) std::memset(t.impl->storage->grad.get(), 0, nbytes);
    }
}


// Helper: create a tensor whose DATA is copied from self.grad
// ------------------------------------------------------------
Tensor tensor_from_grad(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("tensor_from_grad: missing grad buffer");

    // Create a new tensor for the gradient
    Tensor grad_tensor(self.shape(), self._dtype(), false);
    size_t n = self.numel_();
    
    // --- THIS IS A STRIDE-AWARE COPY ---
    // We iterate over `self` (the source) using its multi-dim index
    // and write to `grad_tensor` (the destination) using a flat index,
    // because we know `grad_tensor` is new and contiguous.
    std::vector<size_t> idx_vec(self.impl->ndim, 0);

    for (size_t flat_dest = 0; flat_dest < n; ++flat_dest) {
        // 1. Convert flat destination index to multi-dim index
        size_t rem = flat_dest;
        for (int d = (int)self.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % self.impl->shape[d];
            rem /= self.impl->shape[d];
        }

        // 2. Convert multi-dim index to strided source index
        size_t strided_src_idx = self.impl->offset;
        for (size_t d = 0; d < self.impl->ndim; ++d) {
            strided_src_idx += idx_vec[d] * self.impl->strides[d];
        }
        
        // 3. Read from strided source, write to flat destination
        double gv = read_scalar_at(self.impl->storage->grad.get(), strided_src_idx, self._dtype());
        write_scalar_at(grad_tensor.impl->storage->data.get(), flat_dest, grad_tensor._dtype(), gv);
    }

    return grad_tensor;
}

// copy .data -> .grad (allocate grad buffer and copy values)
// after calling this the gradient values are available in impl->storage->grad
static void copy_data_to_grad(Tensor &t) {
    if (!t.impl) throw std::runtime_error("copy_data_to_grad: undefined tensor");
    size_t n = t.numel();
    ensure_grad_buffer(t, false); // Ensure buffer exists, don't zero it

    // --- THIS MUST BE A STRIDE-AWARE COPY ---
    // We read from .data using strides and write to .grad using strides.
    std::vector<size_t> idx_vec(t.impl->ndim, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // 1. Convert flat index to multi-dim index
        size_t rem = flat;
        for (int d = (int)t.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % t.impl->shape[d];
            rem /= t.impl->shape[d];
        }

        // 2. Convert multi-dim index to strided index (applies to both data and grad)
        size_t strided_idx = t.impl->offset;
        for (size_t d = 0; d < t.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * t.impl->strides[d];
        }
        
        // 3. Read from strided data, write to strided grad
        double v = read_scalar_at(t.impl->storage->data.get(), strided_idx, t.impl->dtype);
        write_scalar_at(t.impl->storage->grad.get(), strided_idx, t.impl->dtype, v);
    }
}

// reduce `t` by summing over axes (axes are indices relative to current ndim).
// After each reduction we unsqueeze at same axis to keep the same ndim but with size 1 in reduced axes.
// This produces a tensor with shape = original shape but with reduced axes set to 1.
static Tensor reduce_sum_axes_keepdims(Tensor t, std::vector<int> axes) {
    if (axes.empty()) return t;
    // sort axes ascending (we will operate in given order but unsqueeze after each sum)
    std::sort(axes.begin(), axes.end());
    // To avoid index shifting issues we will perform reductions left-to-right but unsqueeze immediately.
    for (int ax : axes) {
        // call your sum(t, dim) which removes dimension `ax`
        Tensor s = sum(t, ax); // s has ndim = t.ndim - 1
        // unsqueeze at ax to restore singleton dim
        s = s.unsqueeze(ax); // requires your unsqueeze implemented earlier
        t = s;
    }
    return t;
}

// fetch dimension value of `target` as if left-padded to `nd` dimensions
static size_t dim_in_padded(const Tensor& target, size_t nd, size_t idx) {
    size_t tnd = target.impl->ndim;
    if (idx < nd - tnd) return 1;
    return target.impl->shape[idx - (nd - tnd)];
}

// ------------------ accumulate_grad (broadcast-aware) ------------------
// accumulates gradient from grad_src into target.
// grad_src may have its values in .impl->storage->grad (preferred) or in .impl->storage->data
inline void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) throw std::runtime_error("accumulate_grad: target undefined");
    if (!grad_src.impl) throw std::runtime_error("accumulate_grad: grad_src undefined");
    // Fast-path: accumulate grad for bias-like case:
    // target: [out_c] (ndim==1)
    // grad_src: [batch, out_c, out_w] (ndim==3) -> reduce axes 0 and 2
    if (target.impl && grad_src.impl &&
        target.impl->ndim == 1 &&
        grad_src.impl->ndim == 3 &&
        target.impl->shape[0] == grad_src.impl->shape[1])
    {
        // Ensure grad buffer exists (zero it if newly allocated)
        bool had_gradbuf = (target.impl->storage && target.impl->storage->grad != nullptr);
        ensure_grad_buffer(target, !had_gradbuf);

        void* dst_grad = target.impl->storage->grad.get();
        // Prefer reading from grad buffer of grad_src if present (rare),
        // otherwise read from user-visible data.
        void* src_ptr = (grad_src.impl->storage && grad_src.impl->storage->grad)
                        ? grad_src.impl->storage->grad.get()
                        : grad_src.impl->storage->data.get();

        size_t out_c = target.impl->shape[0];
        size_t batch = grad_src.impl->shape[0];
        size_t out_w = grad_src.impl->shape[2];

        size_t src_off = grad_src.impl->offset;
        size_t dst_off = target.impl->offset;

        size_t s_stride_b = grad_src.impl->strides[0];
        size_t s_stride_oc = grad_src.impl->strides[1];
        size_t s_stride_w = grad_src.impl->strides[2];
        size_t t_stride = target.impl->strides[0];

        DType src_dt = grad_src._dtype();
        DType dst_dt = target._dtype();

        for (size_t oc = 0; oc < out_c; ++oc) {
            double sum = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                for (size_t w = 0; w < out_w; ++w) {
                    size_t idx = src_off + b * s_stride_b + oc * s_stride_oc + w * s_stride_w;
                    sum += read_scalar_at(src_ptr, idx, src_dt);
                }
            }
            size_t ti = dst_off + oc * t_stride;
            double cur = read_scalar_at(dst_grad, ti, dst_dt);
            write_scalar_at(dst_grad, ti, dst_dt, cur + sum);
        }

        // done for this special case
        return;
    }

    size_t nd_t = target.impl->ndim;
    size_t nd_g = grad_src.impl->ndim;

    // Determine axes where target has 1 and grad has >1 (these axes need reduction).
    std::vector<int> axes_to_reduce;
    size_t max_ndim = std::max(nd_t, nd_g);
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t td = dim_in_padded(target, max_ndim, i);
        size_t gd = dim_in_padded(grad_src, max_ndim, i);
        if (td == 1 && gd > 1) {
            axes_to_reduce.push_back((int)(i - (max_ndim - nd_g)));
        }
    }

    // Align grad_src to target shape (reduce broadcast axes and keep dims)
    Tensor g_aligned = grad_src;
    if (!axes_to_reduce.empty()) {
        g_aligned = reduce_sum_axes_keepdims(grad_src, axes_to_reduce);
    }

    // Ensure target grad buffer exists. ZERO it only if we are allocating it now.
    bool had_gradbuf = (target.impl->storage && target.impl->storage->grad != nullptr);
    ensure_grad_buffer(target, !had_gradbuf); // zero only on first allocation

    size_t N = target.numel_();
    if (N == 0) return;

    // choose pointers (read g_aligned from its grad buffer if present, else from data)
    bool g_has_gradbuf = (g_aligned.impl->storage && g_aligned.impl->storage->grad != nullptr);
    void* g_data_ptr = g_has_gradbuf ? g_aligned.impl->storage->grad.get() : g_aligned.impl->storage->data.get();
    void* t_grad_ptr = target.impl->storage->grad.get();

    // multi-dim index vector for the target
    std::vector<size_t> idx_vec(target.impl->ndim, 0);
    
    for (size_t flat = 0; flat < N; ++flat) {
        // convert flat -> multi-index for target
        size_t rem = flat;
        for (int d = (int)target.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % target.impl->shape[d];
            rem /= target.impl->shape[d];
        }

        // target strided index
        size_t target_strided_idx = target.impl->offset;
        for (size_t d = 0; d < target.impl->ndim; ++d) {
            target_strided_idx += idx_vec[d] * target.impl->strides[d];
        }

        // g_aligned strided index (handle left-pad when ndim differ)
        size_t g_aligned_strided_idx = g_aligned.impl->offset;
        size_t pad = target.impl->ndim > g_aligned.impl->ndim ? (target.impl->ndim - g_aligned.impl->ndim) : 0;
        for (size_t d = 0; d < target.impl->ndim; ++d) {
            if (d < pad) continue;
            size_t g_dim_idx = d - pad;
            size_t g_dim_shape = g_aligned.impl->shape[g_dim_idx];
            size_t use_idx = (g_dim_shape == 1) ? 0 : idx_vec[d];
            g_aligned_strided_idx += use_idx * g_aligned.impl->strides[g_dim_idx];
        }

        // debug bounds checks (safe and helpful)
        size_t target_storage_size = target.impl->storage ? target.impl->storage->size : 0;
        size_t g_storage_size = g_aligned.impl->storage ? g_aligned.impl->storage->size : 0;
        if (target_strided_idx >= target_storage_size) {
            throw std::runtime_error("accumulate_grad: target index OOB");
        }
        if (g_aligned_strided_idx >= g_storage_size) {
            throw std::runtime_error("accumulate_grad: g_aligned index OOB");
        }

        // accumulate: t.grad[idx] += g_aligned[idx_broadcasted]
        double addv = read_scalar_at(g_data_ptr, g_aligned_strided_idx, g_aligned._dtype());
        double cur  = read_scalar_at(t_grad_ptr, target_strided_idx, target._dtype());
        write_scalar_at(t_grad_ptr, target_strided_idx, target._dtype(), cur + addv);
    }
}


// ------------------ Backward nodes (use ops1 names) ------------------

// Add

// Sub (diff_)

GradSub::GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradSub::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSub: missing self grad");
    if (a.requires_grad()) accumulate_grad(a, self);
    if (b.requires_grad()) {
        // negated grad: create neg = -self.grad
        Tensor grad_self = tensor_from_grad(self); // Stride-aware copy
        Tensor neg = grad_self * -1.0;             // Use op overload
        
        // copy neg.data to a new grad tensor's .data buffer
        // (accumulate_grad reads from .data if .grad is null)
        accumulate_grad(b, neg);
    }
}


// Mul

GradMul::GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradMul::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradMul: missing self grad");

    // Create grad_self whose DATA holds incoming gradient values
    Tensor grad_self = tensor_from_grad(self); // This is now stride-aware 

    // ga = grad_self * b
    if (a.requires_grad()) {
        bool old_ra = a.impl->requires_grad;
        bool old_rb = b.impl->requires_grad;
        a.impl->requires_grad = false;
        b.impl->requires_grad = false;
        Tensor ga = grad_self * b; // Use op overload
        accumulate_grad(a, ga);
        a.impl->requires_grad = old_ra;
        b.impl->requires_grad = old_rb;
    }

    // gb = grad_self * a
    if (b.requires_grad()) {
        bool old_ra = a.impl->requires_grad;
        bool old_rb = b.impl->requires_grad;
        a.impl->requires_grad = false;
        b.impl->requires_grad = false;
        Tensor gb = grad_self * a; // Use op overload
        accumulate_grad(b, gb);
        a.impl->requires_grad = old_ra;
        b.impl->requires_grad = old_rb;
    }
}


// Div
GradDiv::GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradDiv::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradDiv: missing self grad");

    Tensor grad_self = tensor_from_grad(self); // Stride-aware 

    // ---- In-place “detach” ----
    bool old_grad_a = a.impl->requires_grad;
    bool old_grad_b = b.impl->requires_grad;

    a.impl->requires_grad = false;
    b.impl->requires_grad = false;

    // ---- Compute gradients ----
    if (old_grad_a) {
        Tensor da = grad_self / b;  // grad_self / b
        accumulate_grad(a, da);
    }
    if (old_grad_b) {
        Tensor num = grad_self * a;   // grad_self * a
        Tensor den = b * b;           // b * b
        Tensor db = (num / den) * -1.0; // (grad * a) / (b * b) * -1

        accumulate_grad(b, db);
    }

    // ---- Restore flags ----
    a.impl->requires_grad = old_grad_a;
    b.impl->requires_grad = old_grad_b;
}

GradMatMul::GradMatMul(const Tensor& a_, const Tensor& b_)
    : a(a_), b(b_) {
    parents = {a, b};
}

void GradMatMul::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMatMul: missing self grad");

    // dL/dY (as contiguous DATA tensor)
    Tensor grad_y = tensor_from_grad(self);

    // helper to transpose the last two dims (no-op for ndim < 2)
    auto transpose_last_two = [](const Tensor &t) -> Tensor {
        if (!t.impl) throw std::runtime_error("transpose_last_two: undefined tensor");
        if (t.impl->ndim < 2) return t.clone();
        std::vector<size_t> perm(t.impl->ndim);
        for (size_t i = 0; i < t.impl->ndim; ++i) perm[i] = i;
        std::swap(perm[t.impl->ndim - 2], perm[t.impl->ndim - 1]);
        return t.permute(perm);
    };

    // ----- grad w.r.t. a -----
    if (a.requires_grad()) {
        // vector-vector -> scalar special-case
        if (a.impl->ndim == 1 && b.impl->ndim == 1) {
            // grad_y is scalar
            Tensor da = grad_y * b;        // shape (k,)
            accumulate_grad(a, da);
        } else {
            bool old_ra = a.impl->requires_grad;
            bool old_rb = b.impl->requires_grad;
            a.impl->requires_grad = false;
            b.impl->requires_grad = false;
            // general: dA = dY @ B^T
            Tensor bt = transpose_last_two(b);
            Tensor grad_a = matmul_(grad_y, bt); // shape: batchShape + [m, k]
            a.impl->requires_grad = old_ra;
            b.impl->requires_grad = old_rb;

            // if original `a` was 1-D, reduce leading/batch dims so result becomes shape (k,)
            if (a.impl->ndim == 1) {
                int nd = (int)grad_a.impl->ndim;
                // sum over all axes except the last one
                if (nd > 1) {
                    std::vector<int> axes;
                    for (int ax = 0; ax < nd - 1; ++ax) axes.push_back(ax);
                    grad_a = reduce_sum_axes_keepdims(grad_a, axes); // now shape [..., 1, k] with many 1s
                }
                // collapse to 1-D [k]
                std::vector<size_t> final_shape = { grad_a.impl->shape[grad_a.impl->ndim - 1] };
                grad_a = grad_a.reshape(final_shape);
            }

            // accumulate (accumulate_grad will handle broadcasting/reductions if shapes differ)
            accumulate_grad(a, grad_a);
        }
    }

    // ----- grad w.r.t. b -----
    if (b.requires_grad()) {
        // vector-vector -> scalar special-case
        if (a.impl->ndim == 1 && b.impl->ndim == 1) {
            Tensor db = grad_y * a; // shape (k,)
            accumulate_grad(b, db);
        } else {
            bool old_ra = a.impl->requires_grad;
            bool old_rb = b.impl->requires_grad;
            a.impl->requires_grad = false;
            b.impl->requires_grad = false;
            // general: dB = A^T @ dY
            Tensor at = transpose_last_two(a);
            Tensor grad_b = matmul_(at, grad_y); // shape: batchShape + [k, n]
            a.impl->requires_grad = old_ra;
            b.impl->requires_grad = old_rb;
            // if original `b` was 1-D, reduce leading/batch dims so result becomes shape (k,)
            if (b.impl->ndim == 1) {
                int nd = (int)grad_b.impl->ndim;
                // sum over all axes except the first one (k is at last-1 position when shape is [k,1] etc.)
                if (nd > 1) {
                    std::vector<int> axes;
                    // We want to reduce every axis except the one corresponding to 'k'.
                    // For grad_b shape batchShape + [k, n] but for vector b we expect k dimension to be
                    // at position (nd - 2). Simpler: sum all axes except the one that will map to k.
                    // Build axes list = 0 .. (nd-1) excluding (nd-2)
                    int keep = nd - 2;
                    for (int ax = 0; ax < nd; ++ax) {
                        if (ax == keep) continue;
                        axes.push_back(ax);
                    }
                    grad_b = reduce_sum_axes_keepdims(grad_b, axes); // keeps shape with 1 at reduced axes
                }
                // collapse to 1-D [k]
                std::vector<size_t> final_shape = { grad_b.impl->shape[grad_b.impl->ndim - 2] };
                grad_b = grad_b.reshape(final_shape);
            }

            accumulate_grad(b, grad_b);
        }
    }
}



// Pow elementwise: z = a^b (both tensors)
// da = b * a^(b-1) * grad_self
// db = a^b * ln(a) * grad_self
GradPow::GradPow(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradPow::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradPow: missing self grad");

    Tensor grad_y = tensor_from_grad(self); // Stride-aware

    // 2️⃣ ∂Loss/∂a = grad_y * b * a^(b-1)
    if (a.requires_grad()) {
        // Use ops for broadcasting
        bool old_grad_a = a.requires_grad();
        bool old_grad_b = b.requires_grad();
        a.impl->requires_grad = false;
        b.impl->requires_grad = false;

        Tensor grad_a = grad_y * b * pow_(a, b - 1.0); // <-- FIX #2: Correct derivative math
        
        a.impl->requires_grad = old_grad_a;
        b.impl->requires_grad = old_grad_b;
        
        accumulate_grad(a, grad_a);
    }

    // 3️⃣ ∂Loss/∂b = grad_y * a^b * ln(a)
    if (b.requires_grad()) {
        bool old_grad_a = a.requires_grad();
        bool old_grad_b = b.requires_grad();
        a.impl->requires_grad = false;
        b.impl->requires_grad = false;

        Tensor grad_b = grad_y * pow_(a, b) * ln_(a);
        
        a.impl->requires_grad = old_grad_a;
        b.impl->requires_grad = old_grad_b;
        
        accumulate_grad(b, grad_b);
    }
}

//Scalar
void GradAddScalar::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradAddScalar: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    // derivative wrt a is 1 → just pass grad_self
    accumulate_grad(a, grad_input);
}

void GradSubScalar::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSubScalar: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    // derivative wrt a is 1 (since out = a - scalar)
    accumulate_grad(a, grad_input);
}

void GradSubAfterScalar::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSubAfterScalar: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    // derivative wrt a is -1 (since out = scalar - a)
    grad_input = grad_input * (-1.0);
    accumulate_grad(a, grad_input);
}
void GradMulScalar::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMulScalar: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    grad_input = grad_input * scalar;  // reuse existing op
    accumulate_grad(a, grad_input);
}

void GradLn::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradLn: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;
    
    Tensor grad_ln = grad_input / t; // grad_self / t
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_ln);
}
void GradExp::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradExp: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);

    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_exp = grad_input * exp_(t); // grad_self * exp(t)
    
    t.impl->requires_grad = old_grad_t;
    
    accumulate_grad(t, grad_exp);
}
void GradSqrt::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSqrt: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_sqrt = grad_input / (sqrt_(t) * 2.0); // grad_self / (2 * sqrt(t))
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_sqrt);
}

void GradSin::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSin: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_sin = grad_input * cos_(t); // grad_self * cos(t)
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_sin);
}

void GradASin::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradASin: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);

    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // 1.0 / sqrt(1.0 - t*t)
    Tensor deriv = pow_scalar(1.0 - (t * t), -0.5); // <-- FIX: Was scalar_pow
    Tensor grad_asin = grad_input * deriv;
    
    t.impl->requires_grad = old_grad_t;
    
    accumulate_grad(t, grad_asin);
}
void GradSinH::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSinH: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_sin = grad_input * cosh_(t); // grad_self * cos(t)
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_sin);
}

void GradCos::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradCos: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_cos = grad_input * sin_(t) * -1.0; // grad_self * -sin(t)
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_cos);
}

void GradACos::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradACos: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;
    
    // -1.0 / sqrt(1.0 - t*t)
    Tensor deriv = pow_scalar(1.0 - (t * t), -0.5) * -1.0; // <-- FIX: Was scalar_pow
    Tensor grad_acos = grad_input * deriv;

    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_acos);
}

void GradCosH::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradCosH: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    Tensor grad_cos = grad_input * sinh_(t); // grad_self * -sin(t)
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_cos);
}

void GradTan::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradTan: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // 1.0 / (cos(t) * cos(t))
    Tensor cos_t = cos_(t);
    Tensor deriv = 1.0 / (cos_t * cos_t);
    Tensor grad_tan = grad_input * deriv;
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_tan);
}
void GradATan::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradATan: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // 1.0 / (1.0 + t*t)
    Tensor deriv = 1.0 / (1.0 + (t * t));
    Tensor grad_atan = grad_input * deriv;

    t.impl->requires_grad = old_grad_t;
    
    accumulate_grad(t, grad_atan);
}
void GradTanH::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradTanH: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // 1.0 - tanh(t)^2
    Tensor tanh_t = tanh_(t);
    Tensor deriv = 1.0 - (tanh_t * tanh_t);
    Tensor grad_tanh = grad_input * deriv;

    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_tanh);
}

void GradSigmoid::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("Sigmoid: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;
    
    // We need the *output* of sigmoid, which is `self`
    // deriv = self * (1.0 - self)
    Tensor deriv = self * (1.0 - self);
    Tensor grad_sig = grad_input * deriv;

    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_sig);
}
void GradRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("Relu: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // deriv = 1.0 if t > 0 else 0.0
    Tensor deriv = gt(t, 0.0); // gt(t, 0.0) returns 1.0 or 0.0
    Tensor grad_relu = grad_input * deriv;
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_relu);
}
void GradSoftPlus::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("SoftPlus: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);

    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;

    // deriv = 1.0 / (1.0 + exp(-t)) (which is just sigmoid(t))
    Tensor deriv = sigmoid_(t);
    Tensor grad_sp = grad_input * deriv;
    
    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_sp);
}

// ------------------ topo sort helper ------------------
static void topo_sort_from(const Tensor& root, std::vector<Tensor>& topo) {
    std::set<const Tensorimpl*> visited;
    std::function<void(const Tensor&)> dfs = [&](const Tensor& t){
        if (!t.impl) return;
        const Tensorimpl* ip = t.impl.get();
        if (visited.count(ip)) return;
        visited.insert(ip);

        if (t.impl->grad_fn) {
            for (const Tensor& p : t.impl->grad_fn->parents) dfs(p);
        }
        topo.push_back(t);
    };
    dfs(root);
}

void GradAbs::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradAbs: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    
    // Use ops for broadcasting
    bool old_grad_t = t.requires_grad();
    t.impl->requires_grad = false;
    
    // deriv = 1.0 if t > 0, -1.0 if t < 0, 0.0 if t == 0
    Tensor deriv = gt(t, 0.0) - lt(t, 0.0);
    Tensor grad_abs = grad_input * deriv;

    t.impl->requires_grad = old_grad_t;

    accumulate_grad(t, grad_abs);
}

void GradSum::backward(const Tensor& self) {
    // self.impl->storage->grad must exist (scalar gradient)
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSum: missing self grad");

    if (!t.impl || !t.requires_grad()) return;

    // scalar gradient value (assume scalar stored at index 0, check for view)
    double g = read_scalar_at(self.impl->storage->grad.get(), self.impl->offset, self.impl->dtype); // <-- FIX #3: Use offset

    // create grad tensor of same shape as t and fill with g
    std::vector<size_t> shape_vec(t.impl->shape, t.impl->shape + t.impl->ndim);
    Tensor grad_input = Tensor::full(shape_vec, g, t.impl->dtype, false);

    // accumulate into t's grad storage via existing helper
    accumulate_grad(t, grad_input);
}
void GradMean::backward(const Tensor& self) {
    // self.impl->storage->grad must exist (scalar gradient)
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMean: missing self grad");

    if (!t.impl || !t.requires_grad()) return;

    // scalar gradient value (assume scalar stored at index 0, check for view)
    double g = read_scalar_at(self.impl->storage->grad.get(), self.impl->offset, self.impl->dtype); // <-- FIX #3: Use offset

    // mean divides by number of elements
    size_t n = t.numel_();
    double scaled_g = g / static_cast<double>(n);

    // create grad tensor of same shape as t and fill with scaled_g
    std::vector<size_t> shape_vec(t.impl->shape, t.impl->shape + t.impl->ndim);
    Tensor grad_input = Tensor::full(shape_vec, scaled_g, t.impl->dtype, false);

    // accumulate into t's grad storage via existing helper
    accumulate_grad(t, grad_input);
}

// ------------------ backward ------------------
void backward(Tensor& loss) {
    if (!loss.impl) throw std::runtime_error("backward: loss undefined");
    if (!loss.impl->requires_grad) throw std::runtime_error("backward: loss requires_grad == false");

    // build topo order first (so we know which tensors are in the graph)
    std::vector<Tensor> topo;
    topo_sort_from(loss, topo);

    // Zero (or allocate+zero) grad buffers for all tensors in the graph that may receive grads.
    // This prevents accumulation from previous backward calls or uninitialized memory.
    for (Tensor &t : topo) {
        if (!t.impl) continue;
        // only zero buffers for tensors that require_grad (leaf or param tensors)
        if (t.impl->requires_grad) {
            ensure_grad_buffer(t, true); // allocate if needed and zero entire underlying storage
        }
    }

    // Now set loss grad to ones (stride-aware)
    // Note: loss may be in topo, we already zeroed its grad buffer above.
    size_t n = loss.numel();
    std::vector<size_t> idx_vec(loss.impl->ndim, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        for (int d = (int)loss.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % loss.impl->shape[d];
            rem /= loss.impl->shape[d];
        }
        size_t strided_idx = loss.impl->offset;
        for (size_t d = 0; d < loss.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * loss.impl->strides[d];
        }
        write_scalar_at(loss.impl->storage->grad.get(), strided_idx, loss._dtype(), 1.0);
    }

    // run backwards in reverse topo order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor cur = *it;
        if (!cur.impl->grad_fn) continue;
        cur.impl->grad_fn->backward(cur);
    }
}
Tensor grad_of(const Tensor& t) {
    if (!t.impl || !t.impl->requires_grad)
        return Tensor(); // no gradient

    if (!t.impl->storage || !t.impl->storage->grad)
        return Tensor(); // no gradient computed

    Tensor g;
    g.impl = std::make_shared<Tensorimpl>(
        t.shape(),              // same shape
        t.impl->dtype,          // same dtype
        false                   // gradients of gradients? no
    );

    // Reuse same storage (important!)
    g.impl->storage = t.impl->storage;

    // Same offset, shape, strides (view)
    g.impl->offset = t.impl->offset;
    g.impl->ndim = t.impl->ndim;

    // copy shape
    for (size_t i = 0; i < t.impl->ndim; i++)
        g.impl->shape[i] = t.impl->shape[i];

    // copy strides
    for (size_t i = 0; i < t.impl->ndim; i++)
        g.impl->strides[i] = t.impl->strides[i];

    // Important: do not attach a grad_fn to a pure gradient
    g.impl->grad_fn = nullptr;

    return g;
}
