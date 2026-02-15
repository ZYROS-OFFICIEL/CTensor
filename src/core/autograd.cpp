#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <set>
#include <functional>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <atomic> // Needed for thread-safe warning flag

#include "autograd.h" 
#include "tensor.h"

//-------------------- helpers --------------------
void ensure_grad_buffer(Tensor &t, bool zero_existing) {
    if (!t.impl) throw std::runtime_error("ensure_grad_buffer: undefined tensor");

    // Allocate if missing
    if (!t.impl->grad) {
        t.impl->grad = intrusive_ptr<Tensorimpl>(new Tensorimpl(
            t.shape(), 
            t._dtype(), 
            false, 
            t.device()
        ));
        
        // Zero it out
        size_t nbytes = t.numel() * t.dtype_bytes();
        if (nbytes > 0 && t.impl->grad->data && t.impl->grad->data->data) {
            std::memset(t.impl->grad->data->data.get(), 0, nbytes);
        }
    } 
    // Zero existing if requested
    else if (zero_existing) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        if (nbytes > 0 && t.impl->grad->data && t.impl->grad->data->data) {
            std::memset(t.impl->grad->data->data.get(), 0, nbytes);
        }
    }
}

// Convert the raw gradient buffer into a usable Tensor object
Tensor tensor_from_grad(const Tensor& self) {
    if (!self.impl || !self.impl->grad)
        throw std::runtime_error("tensor_from_grad: missing grad buffer");

    Tensor g;
    g.impl = self.impl->grad; // This increments ref count
    return g;
}

// --- Accumulation Logic ---

// Helper to check padding logic
static size_t dim_padded(const Tensor& t, size_t max_dims, size_t i) {
    size_t t_dims = t.impl->ndim;
    if (i < max_dims - t_dims) return 1;
    return t.impl->shape[i - (max_dims - t_dims)];
}

// Static flag to warn about NaNs only once
static std::atomic<bool> nan_warned{false};

void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) return; 

    // 1. Identify dimensions to reduce (Broadcasting backward rule)
    size_t t_dims = target.impl->ndim;
    size_t g_dims = grad_src.impl->ndim;
    size_t max_dims = std::max(t_dims, g_dims);
    
    std::vector<int> axes_to_reduce;
    for (size_t i = 0; i < max_dims; ++i) {
        size_t t_sz = dim_padded(target, max_dims, i);
        size_t g_sz = dim_padded(grad_src, max_dims, i);
        
        if (t_sz == 1 && g_sz > 1) {
            // Must reduce this axis. Calculate actual axis index in grad_src
            if (i >= max_dims - g_dims) {
                axes_to_reduce.push_back((int)(i - (max_dims - g_dims)));
            }
        }
    }

    // 2. Perform Reduction (if needed)
    Tensor grad_aligned = grad_src;
    if (!axes_to_reduce.empty()) {
        for (int ax : axes_to_reduce) {
            grad_aligned = sum(grad_aligned, ax);
        }
    }

    // 3. Add to target.grad
    ensure_grad_buffer(target, false);
    
    size_t n_target = target.numel();
    auto* t_grad = target.impl->grad->data->data.get();
    auto* g_data = grad_aligned.impl->data->data.get();
    
    const size_t* t_shape = target.impl->shape.data();
    
    // --- FIX: Use GRAD strides/offset, not Tensor strides/offset ---
    // The gradient buffer is a separate tensor that might be contiguous 
    // even if the target is a view (e.g., transposed).
    const size_t* t_strides = target.impl->grad->strides.data();
    size_t t_offset = target.impl->grad->offset; 

    const size_t* g_shape = grad_aligned.impl->shape.data();
    const size_t* g_strides = grad_aligned.impl->strides.data();
    
    size_t g_offset = grad_aligned.impl->offset;
    
    size_t t_nd = target.impl->ndim;
    size_t pad = (t_nd > grad_aligned.impl->ndim) ? t_nd - grad_aligned.impl->ndim : 0;
    
    DType dt_t = target._dtype();
    DType dt_g = grad_aligned._dtype();

    #pragma omp parallel for
    for (size_t i = 0; i < n_target; ++i) {
        size_t rem = i;
        size_t t_idx = t_offset;
        size_t g_idx = g_offset;
        
        // Calculate coords based on TARGET SHAPE (Logical)
        // But map to memory using GRAD STRIDES (Physical destination)
        for (int d = (int)t_nd - 1; d >= 0; --d) {
            size_t sz = t_shape[d];
            size_t coord = rem % sz;
            rem /= sz;
            
            t_idx += coord * t_strides[d];
            
            // Map to grad_aligned source
            if (d >= (int)pad) {
                int g_d = d - pad;
                if (g_d < (int)grad_aligned.impl->ndim && g_shape[g_d] > 1) {
                    g_idx += coord * g_strides[g_d];
                }
            }
        }
        
        double v_g = read_scalar_at(g_data, g_idx, dt_g);
        
        // --- SAFETY CHECK: Ignore NaNs/Infs ---
        if (!std::isfinite(v_g)) {
            if(!nan_warned.exchange(true)) {
                std::cerr << "Warning: NaN/Inf gradient detected and suppressed in accumulate_grad.\n";
            }
            continue; // Skip this gradient contribution
        }

        double v_t = read_scalar_at(t_grad, t_idx, dt_t);
        write_scalar_at(t_grad, t_idx, dt_t, v_t + v_g);
    }
}

// ------------------ GradFn implementations ------------------

// Constructors for Grad structs 
GradSub::GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
GradMul::GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
GradDiv::GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
GradPow::GradPow(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
GradMatMul::GradMatMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }

void GradAdd::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) {
        accumulate_grad(a, grad);
    }
    if (b.requires_grad()) { 
        accumulate_grad(b, grad);
    }
}

void GradSub::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) accumulate_grad(a, grad);
    if (b.requires_grad()) accumulate_grad(b, mul_scalar(grad, -1.0));
}

void GradMul::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) accumulate_grad(a, mul(grad, b)); 
    if (b.requires_grad()) accumulate_grad(b, mul(grad, a));
}
void GradDiv::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) {
        accumulate_grad(a, div(grad, b));
    }
    if (b.requires_grad()) {
        Tensor num = mul(grad, a); 
        Tensor den = mul(b, b);    
        // FIX: Add stability to denominator to prevent NaN
        // Increased epsilon to 1e-6 for float32 safety
        Tensor den_safe = add_scalar(den, 1e-6); 
        Tensor res = div(num, den_safe);
        accumulate_grad(b, mul_scalar(res, -1.0));
    }
}

void GradMatMul::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    
    //transpose
    auto T = [](const Tensor& t) {
        if (t.impl->ndim < 2) return t;
        std::vector<size_t> perm(t.impl->ndim);
        for(size_t i=0; i<t.impl->ndim; ++i) perm[i]=i;
        std::swap(perm[t.impl->ndim-1], perm[t.impl->ndim-2]);
        return t.permute(perm);
    };

    if (a.requires_grad()) {
        accumulate_grad(a, matmul(grad, T(b)));
    }
    if (b.requires_grad()) {
        accumulate_grad(b, matmul(T(a), grad));
    }
}

void GradPow::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) {
        Tensor t1 = pow(a, sub_scalar(b, 1.0));
        Tensor t2 = mul(t1, b);
        accumulate_grad(a, mul(grad, t2));
    }
    if (b.requires_grad()) {
        Tensor t1 = pow(a, b);
        Tensor t2 = log(a);
        Tensor t3 = mul(t1, t2);
        accumulate_grad(b, mul(grad, t3));
    }
}

//--------------------Scalar backward --------------------
void GradAddScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, tensor_from_grad(self));
}
void GradSubScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, tensor_from_grad(self));
}
void GradSubAfterScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, mul_scalar(tensor_from_grad(self), -1.0));
}
void GradMulScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, mul_scalar(tensor_from_grad(self), scalar));
}
void GradDivScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, div_scalar(tensor_from_grad(self), s));
}

void GradScalarDiv::backward(const Tensor& self) {
    if (a.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        Tensor den = mul(a, a);
        Tensor val = div_scalar_rev(s, den); // s / a^2
        Tensor res = mul(grad, val);
        accumulate_grad(a, mul_scalar(res, -1.0));
    }
}

void GradPowScalar::backward(const Tensor& self) {
    if (a.requires_grad()) {
        // y = a^s -> dy/da = s * a^(s-1)
        Tensor grad = tensor_from_grad(self);
        Tensor p = pow_scalar(a, s - 1.0);
        Tensor res = mul_scalar(p, s);
        accumulate_grad(a, mul(grad, res));
    }
}

void GradScalarPow::backward(const Tensor& self) {
    if (a.requires_grad()) {
        // y = s^a -> dy/da = s^a * ln(s)
        Tensor grad = tensor_from_grad(self);
        Tensor p = pow_scalar_rev(scalar, a);
        Tensor res = mul_scalar(p, std::log(scalar));
        accumulate_grad(a, mul(grad, res));
    }
}

//-------------------- Unary backward --------------------
void GradAbs::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        // sign(t) * grad.  (Using ge(0) - lt(0))
        Tensor pos = gt(t, 0.0);
        Tensor neg = lt(t, 0.0);
        Tensor sign = sub(pos, neg); // 1 if >0, -1 if <0
        accumulate_grad(t, mul(grad, sign));
    }
}
void GradLn::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // 1/t * grad
        // FIX: Add epsilon to avoid division by zero (which causes Infinity/NaN gradients)
        Tensor grad = tensor_from_grad(self);
        Tensor t_safe = add_scalar(t, 1e-6);
        accumulate_grad(t, div(grad, t_safe));
    }
}
void GradExp::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        accumulate_grad(t, mul(grad, exp(t)));
    }
}
void GradSqrt::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // 1/(2*sqrt(t)) * grad
        Tensor grad = tensor_from_grad(self);
        // FIX: Add epsilon to avoid division by zero
        Tensor safe_t = add_scalar(t, 1e-6);
        Tensor two_sqrt = mul_scalar(sqrt(safe_t), 2.0);
        accumulate_grad(t, div(grad, two_sqrt));
    }
}
void GradSin::backward(const Tensor& self) {
    if (t.requires_grad()) accumulate_grad(t, mul(tensor_from_grad(self), cos(t)));
}
void GradCos::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor res = mul(tensor_from_grad(self), sin(t));
        accumulate_grad(t, mul_scalar(res, -1.0));
    }
}


void GradTan::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // sec^2(t) = 1/cos^2(t)
        Tensor c = cos(t);
        Tensor c2 = mul(c, c);
        Tensor deriv = div_scalar_rev(1.0, c2);
        accumulate_grad(t, mul(tensor_from_grad(self), deriv));
    }
}

void GradSigmoid::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        Tensor one_minus = sub_scalar_rev(1.0, self); // 1 - y
        Tensor deriv = mul(self, one_minus);
        accumulate_grad(t, mul(grad, deriv));
    }
}

void GradRelu::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        Tensor mask = gt(t, 0.0);
        accumulate_grad(t, mul(grad, mask));
    }
}

void GradSoftplus::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // sigmoid(x)
        Tensor grad = tensor_from_grad(self);
        Tensor sig = sigmoid(t);
        accumulate_grad(t, mul(grad, sig));
    }
}
//--------------------Reduction backward --------------------
void GradSum::backward(const Tensor& self) {
    if (t.requires_grad()) {
        accumulate_grad(t, tensor_from_grad(self));
    }
}

void GradMean::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // grad / N
        Tensor grad = tensor_from_grad(self);
        double N = 1.0;
        if (dim == -1) N = (double)t.numel();
        else N = (double)t.impl->shape[dim];
        
        accumulate_grad(t, div_scalar(grad, N));
    }
}

GradPermute::GradPermute(const Tensor& t_, const std::vector<size_t>& dims) 
    : t(t_), forward_dims(dims) {
    parents = {t};
    reverse_dims.resize(dims.size());
    for(size_t i=0; i<dims.size(); ++i) reverse_dims[dims[i]] = i;
}

void GradPermute::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        accumulate_grad(t, grad.permute(reverse_dims));
    }
}

void GradReshape::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        // Force contiguous copy to ensure safe reshape
        Tensor contig = grad.contiguous();
        accumulate_grad(t, contig.reshape(old_shape));
    }
}
void GradASin::backward(const Tensor& s){ 
    if(t.requires_grad()) {
        accumulate_grad(t, mul(tensor_from_grad(s), pow_scalar(sub_scalar_rev(1.0, mul(t,t)), -0.5))); 
    }
}
void GradACos::backward(const Tensor& s){ 
    if(t.requires_grad()) {
        accumulate_grad(t, mul_scalar(mul(tensor_from_grad(s), pow_scalar(sub_scalar_rev(1.0, mul(t,t)), -0.5)), -1.0)); 
    }
}
void GradATan::backward(const Tensor& s){
    if(t.requires_grad()) {
        accumulate_grad(t, div(tensor_from_grad(s), add_scalar(mul(t,t), 1.0))); 
    }
}
void GradSinh::backward(const Tensor& s){ 
    if(t.requires_grad()) {
        accumulate_grad(t, mul(tensor_from_grad(s), cosh(t))); 
    }
}
void GradCosh::backward(const Tensor& s){ 
    if(t.requires_grad()) {
        accumulate_grad(t, mul(tensor_from_grad(s), sinh(t))); 
    }
}
void GradTanh::backward(const Tensor& s){ 
    if(t.requires_grad()) {
        accumulate_grad(t, mul(tensor_from_grad(s), div_scalar_rev(1.0, mul(cosh(t), cosh(t))))); 
    }
}

// ----------------- GradGather Backward Implementation -----------------
void GradGather::backward(const Tensor& self) {
    if (!t.requires_grad()) return;

    Tensor grad_output = tensor_from_grad(self);
    // Create a zero tensor of the same shape as source input 't'
    Tensor grad_input = Tensor::zeros(t.shape(), t._dtype(), false);
    
    // Scatter Add Logic
    // grad_input[index] += grad_output
    
    // We must iterate over grad_input using 'index' to place gradients back
    Tensor index_cont = index.contiguous();
    Tensor grad_out_cont = grad_output.contiguous();
    
    void* in_ptr = grad_input.impl->data->data.get();
    void* idx_ptr = index_cont.impl->data->data.get();
    void* out_ptr = grad_out_cont.impl->data->data.get();
    
    size_t N = index.numel();
    const auto& inp_shape = t.shape();
    const auto& inp_strides = t.impl->strides;
    
    size_t ndim = t.impl->ndim;
    std::vector<size_t> coords(ndim, 0);

    DType dt = t._dtype();
    DType idx_dt = index._dtype();

    // Iterate over output/index (same shape)
    for (size_t i = 0; i < N; ++i) {
        // 1. Get index value
        double idx_val_raw = read_scalar_at(idx_ptr, i, idx_dt);
        int64_t idx = static_cast<int64_t>(idx_val_raw);
        if (idx < 0) idx += (int64_t)inp_shape[dim];
        
        // 2. Compute offset in grad_input
        size_t inp_offset = 0; 
        for (size_t d = 0; d < ndim; ++d) {
            size_t coord_at_d = (d == dim) ? (size_t)idx : coords[d];
            if (coord_at_d >= inp_shape[d]) continue; // OOB safety
            inp_offset += coord_at_d * inp_strides[d];
        }

        // 3. Read current value at grad_input, add grad_output, write back
        double curr_val = read_scalar_at(in_ptr, inp_offset, dt);
        double incoming_grad = read_scalar_at(out_ptr, i, dt);
        write_scalar_at(in_ptr, inp_offset, dt, curr_val + incoming_grad);

        // 4. Update odometer
        for (int d = (int)ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < index_cont.shape()[d]) break;
            coords[d] = 0;
        }
    }
    
    accumulate_grad(t, grad_input);
}

void backward(Tensor& root) {
    if (!root.impl || !root.requires_grad()) 
        throw std::runtime_error("backward: tensor does not require grad");

    // 1. Initialize Root Grad (1.0)
    ensure_grad_buffer(root, true);
    
    // Fill with 1.0
    auto* g_ptr = root.impl->grad->data->data.get();
    DType dt = root._dtype();
    size_t n = root.numel();
    #pragma omp parallel for
    for(size_t i=0; i<n; ++i) write_scalar_at(g_ptr, i, dt, 1.0);

    // 2. Topo Sort
    std::vector<Tensor> sort;
    std::set<const Tensorimpl*> visited;
    
    std::function<void(const Tensor&)> dfs = [&](const Tensor& u) {
        if (!u.impl || visited.count(u.impl.get())) return;
        visited.insert(u.impl.get());
        if (u.impl->grad_fn) {
            for (const auto& p : u.impl->grad_fn->parents) dfs(p);
        }
        sort.push_back(u);
    };
    dfs(root);

    // 3. Backward Pass (Reverse Topological)
    for (auto it = sort.rbegin(); it != sort.rend(); ++it) {
        Tensor& t = *it;
        if (t.impl->grad_fn) {
            t.impl->grad_fn->backward(t);
        }
    }
}