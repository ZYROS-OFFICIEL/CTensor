#include "autograd.h"
#include "ops.h"       
#include "tensor.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <set>
#include <functional>
#include <iostream>
#include <omp.h>

//-------------------- helpers --------------------
void ensure_grad_buffer(Tensor &t, bool zero_existing) {
    if (!t.impl) throw std::runtime_error("ensure_grad_buffer: undefined tensor");

    // Allocate if missing
    // NOTE: 'grad' is now a shared_ptr<Tensorimpl>, so we create a new Tensorimpl
    if (!t.impl->grad) {
        // Create a new Tensorimpl with same shape/dtype/device
        t.impl->grad = std::make_shared<Tensorimpl>(
            t.shape(), 
            t._dtype(), 
            false, 
            t.device()
        );
        
        // Zero it out
        size_t nbytes = t.numel() * t.dtype_bytes();
        // Access: impl->grad (Tensorimpl) -> data (Storage) -> data (void*)
        if (nbytes > 0 && t.impl->grad->data && t.impl->grad->data->data) {
            std::memset(t.impl->grad->data->data.get(), 0, nbytes);
        }
    } 
    // Zero existing if requested (e.g., at start of backward pass)
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

    // Since 'grad' is now a Tensorimpl, we can just wrap it in a Tensor object.
    // This avoids the manual copy loop entirely!
    Tensor g;
    g.impl = self.impl->grad;
    return g;
}
// --- Accumulation Logic ---

// Helper to check padding logic
static size_t dim_padded(const Tensor& t, size_t max_dims, size_t i) {
    size_t t_dims = t.impl->ndim;
    if (i < max_dims - t_dims) return 1;
    return t.impl->shape[i - (max_dims - t_dims)];
}

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
            grad_aligned = Ops::sum(grad_aligned, ax);
        }
    }

    // 3. Add to target.grad
    ensure_grad_buffer(target, false);
    
    size_t n_target = target.numel();
    
    // Access the destination gradient data
    // target.impl -> grad (Tensorimpl) -> data (Storage) -> data (void*)
    auto* t_grad = target.impl->grad->data->data.get();
    
    // Access the source gradient data
    // grad_aligned.impl -> data (Storage) -> data (void*)
    auto* g_data = grad_aligned.impl->data->data.get();
    
    const size_t* t_shape = target.impl->shape.data();
    const size_t* t_strides = target.impl->strides.data();
    const size_t* g_shape = grad_aligned.impl->shape.data();
    const size_t* g_strides = grad_aligned.impl->strides.data();
    
    size_t t_offset = target.impl->offset; // Usually 0 for gradients, but respecting structure
    size_t g_offset = grad_aligned.impl->offset;
    
    size_t t_nd = target.impl->ndim;
    size_t g_nd = grad_aligned.impl->ndim;
    size_t pad = (t_nd > g_nd) ? t_nd - g_nd : 0;
    
    DType dt_t = target._dtype();
    DType dt_g = grad_aligned._dtype();

    #pragma omp parallel for
    for (size_t i = 0; i < n_target; ++i) {
        size_t rem = i;
        size_t t_idx = t_offset;
        size_t g_idx = g_offset;
        
        for (int d = (int)t_nd - 1; d >= 0; --d) {
            size_t sz = t_shape[d];
            size_t coord = rem % sz;
            rem /= sz;
            
            t_idx += coord * t_strides[d];
            
            // Map to grad_aligned
            if (d >= (int)pad) {
                int g_d = d - pad;
                if (g_shape[g_d] > 1) {
                    g_idx += coord * g_strides[g_d];
                }
            }
        }
        
        double v_g = read_scalar_at(g_data, g_idx, dt_g);
        double v_t = read_scalar_at(t_grad, t_idx, dt_t);
        write_scalar_at(t_grad, t_idx, dt_t, v_t + v_g);
    }
}

// ------------------ GradFn implementations ------------------
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
    if (b.requires_grad()) accumulate_grad(b, Ops::mul_scalar(grad, -1.0));
}

void GradMul::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) accumulate_grad(a, Ops::mul(grad, b)); 
    if (b.requires_grad()) accumulate_grad(b, Ops::mul(grad, a));
}
void GradDiv::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) {
        accumulate_grad(a, Ops::div(grad, b));
    }
    if (b.requires_grad()) {
        Tensor num = Ops::mul(grad, a); 
        Tensor den = Ops::mul(b, b);    
        Tensor res = Ops::div(num, den);
        accumulate_grad(b, Ops::mul_scalar(res, -1.0));
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
        accumulate_grad(a, Ops::matmul(grad, T(b)));
    }
    if (b.requires_grad()) {
        accumulate_grad(b, Ops::matmul(T(a), grad));
    }
}

void GradPow::backward(const Tensor& self) {
    Tensor grad = tensor_from_grad(self);
    if (a.requires_grad()) {
        Tensor t1 = Ops::pow(a, Ops::sub_scalar(b, 1.0));
        Tensor t2 = Ops::mul(t1, b);
        accumulate_grad(a, Ops::mul(grad, t2));
    }
    if (b.requires_grad()) {
        Tensor t1 = Ops::pow(a, b);
        Tensor t2 = Ops::log(a);
        Tensor t3 = Ops::mul(t1, t2);
        accumulate_grad(b, Ops::mul(grad, t3));
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
    if (a.requires_grad()) accumulate_grad(a, Ops::mul_scalar(tensor_from_grad(self), -1.0));
}
void GradMulScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, Ops::mul_scalar(tensor_from_grad(self), s));
}
void GradDivScalar::backward(const Tensor& self) {
    if (a.requires_grad()) accumulate_grad(a, Ops::div_scalar(tensor_from_grad(self), s));
}

void GradScalarDiv::backward(const Tensor& self) {
    if (a.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        Tensor den = Ops::mul(a, a);
        Tensor val = Ops::div_scalar_rev(s, den); // s / a^2
        Tensor res = Ops::mul(grad, val);
        accumulate_grad(a, Ops::mul_scalar(res, -1.0));
    }
}
void GradPowScalar::backward(const Tensor& self) {
    if (a.requires_grad()) {
        // y = a^s -> dy/da = s * a^(s-1)
        Tensor grad = tensor_from_grad(self);
        Tensor p = Ops::pow_scalar(a, s - 1.0);
        Tensor res = Ops::mul_scalar(p, s);
        accumulate_grad(a, Ops::mul(grad, res));
    }
}
void GradScalarPow::backward(const Tensor& self) {
    if (a.requires_grad()) {
        // y = s^a -> dy/da = s^a * ln(s)
        Tensor grad = tensor_from_grad(self);
        Tensor p = Ops::pow_scalar_rev(s, a);
        Tensor res = Ops::mul_scalar(p, std::log(s));
        accumulate_grad(a, Ops::mul(grad, res));
    }
}

//-------------------- Unary backward --------------------
void GradAbs::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        // sign(t) * grad.  (Using ge(0) - lt(0))
        Tensor pos = Ops::gt(t, 0.0);
        Tensor neg = Ops::lt(t, 0.0);
        Tensor sign = Ops::sub(pos, neg); // 1 if >0, -1 if <0
        accumulate_grad(t, Ops::mul(grad, sign));
    }
}
void GradLn::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // 1/t * grad
        Tensor grad = tensor_from_grad(self);
        accumulate_grad(t, Ops::div(grad, t));
    }
}
void GradExp::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // exp(t) * grad -> self * grad (since self IS exp(t))
        // Recomputing exp(t) is safer if self memory reused, but passing self is opt.
        Tensor grad = tensor_from_grad(self);
        accumulate_grad(t, Ops::mul(grad, Ops::exp(t)));
    }
}
void GradSqrt::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // 1/(2*sqrt(t)) * grad
        Tensor grad = tensor_from_grad(self);
        Tensor two_sqrt = Ops::mul_scalar(Ops::sqrt(t), 2.0);
        accumulate_grad(t, Ops::div(grad, two_sqrt));
    }
}
void GradSin::backward(const Tensor& self) {
    if (t.requires_grad()) accumulate_grad(t, Ops::mul(tensor_from_grad(self), Ops::cos(t)));
}
void GradCos::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor res = Ops::mul(tensor_from_grad(self), Ops::sin(t));
        accumulate_grad(t, Ops::mul_scalar(res, -1.0));
    }
}


void GradTan::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // sec^2(t) = 1/cos^2(t)
        Tensor c = Ops::cos(t);
        Tensor c2 = Ops::mul(c, c);
        Tensor deriv = Ops::div_scalar_rev(1.0, c2);
        accumulate_grad(t, Ops::mul(tensor_from_grad(self), deriv));
    }
}

void GradSigmoid::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // sig * (1 - sig). We can use 'self' as sig output.
        // Assuming 'self' hasn't been modified in place.
        // For safety, let's recompute or use 'self' carefully.
        Tensor grad = tensor_from_grad(self);
        Tensor one_minus = Ops::sub_scalar_rev(1.0, self); // 1 - y
        Tensor deriv = Ops::mul(self, one_minus);
        accumulate_grad(t, Ops::mul(grad, deriv));
    }
}

void GradRelu::backward(const Tensor& self) {
    if (t.requires_grad()) {
        Tensor grad = tensor_from_grad(self);
        Tensor mask = Ops::gt(t, 0.0);
        accumulate_grad(t, Ops::mul(grad, mask));
    }
}

void GradSoftplus::backward(const Tensor& self) {
    if (t.requires_grad()) {
        // sigmoid(x)
        Tensor grad = tensor_from_grad(self);
        Tensor sig = Ops::sigmoid(t);
        accumulate_grad(t, Ops::mul(grad, sig));
    }
}
//--------------------Reduction backward --------------------
void GradSum::backward(const Tensor& self) {
    if (t.requires_grad()) {
        accumulate_grad(t, tensor_from_grad(self));
    }
}

void backward(Tensor& root) {
    if (!root.impl || !root.requires_grad()) 
        throw std::runtime_error("backward: tensor does not require grad");

    // 1. Initialize Root Grad (1.0)
    ensure_grad_buffer(root, true);
    
    // Fill with 1.0
    // Access: root.impl -> grad (Tensorimpl) -> data (Storage) -> data (void*)
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
