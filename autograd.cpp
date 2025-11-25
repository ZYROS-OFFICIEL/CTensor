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
#include <omp.h> // OpenMP
#include "autograd.h"
#include "tensor1.h"
#include "opsmp.h" // Use Multi-Threaded Ops by default!

// -------------------- helpers --------------------

// ensure grad buffer exists on tensor; if zero=true fill with zeros
inline void ensure_grad_buffer(Tensor &t, bool zero) {
    if (!t.impl) throw std::runtime_error("ensure_grad_buffer: tensor undefined");
    if (!t.impl->storage->grad) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        void* gptr = std::malloc(nbytes);
        if (!gptr && nbytes) throw std::bad_alloc();
        if (zero) std::memset(gptr, 0, nbytes);
        t.impl->storage->grad = std::shared_ptr<void>(gptr, std::free);
    } else if (zero) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        std::memset(t.impl->storage->grad.get(), 0, nbytes);
    }
}

// Helper: create a tensor whose DATA is copied from self.grad
Tensor tensor_from_grad(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("tensor_from_grad: missing grad buffer");

    Tensor grad_tensor(self.shape(), self._dtype(), false);
    size_t n = self.numel();
    
    // Stride-aware parallel copy
    // Reading from self.grad (strided), writing to grad_tensor.data (contiguous)
    
    size_t ndim = self.impl->ndim;
    const size_t* shape = self.impl->shape;
    const size_t* strides = self.impl->strides;
    size_t offset_base = self.impl->offset;
    DType dt = self._dtype();
    auto* src_grad = self.impl->storage->grad.get();
    auto* dst_data = grad_tensor.impl->storage->data.get();

    #pragma omp parallel for
    for (size_t flat_dest = 0; flat_dest < n; ++flat_dest) {
        size_t rem = flat_dest;
        size_t strided_src_idx = offset_base;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            strided_src_idx += coord * strides[d];
        }
        
        double gv = read_scalar_at(src_grad, strided_src_idx, dt);
        write_scalar_at(dst_data, flat_dest, dt, gv);
    }

    return grad_tensor;
}

// copy .data -> .grad (allocate grad buffer and copy values)
static void copy_data_to_grad(Tensor &t) {
    if (!t.impl) throw std::runtime_error("copy_data_to_grad: undefined tensor");
    size_t n = t.numel();
    ensure_grad_buffer(t, false); 

    size_t ndim = t.impl->ndim;
    const size_t* shape = t.impl->shape;
    const size_t* strides = t.impl->strides;
    size_t offset = t.impl->offset;
    DType dt = t.impl->dtype;
    auto* src_data = t.impl->storage->data.get();
    auto* dst_grad = t.impl->storage->grad.get();

    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t strided_idx = offset;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            strided_idx += coord * strides[d];
        }
        
        double v = read_scalar_at(src_data, strided_idx, dt);
        write_scalar_at(dst_grad, strided_idx, dt, v);
    }
}

// reduce `t` by summing over axes (keeping dims)
// Note: This calls sum_mp recursively, so it benefits from OpenMP automatically.
static Tensor reduce_sum_axes_keepdims(Tensor t, std::vector<int> axes) {
    if (axes.empty()) return t;
    std::sort(axes.begin(), axes.end());
    for (int ax : axes) {
        Tensor s = sum_mp(t, ax); // Using Optimized Sum
        s = s.unsqueeze(ax); 
        t = s;
    }
    return t;
}

static size_t dim_in_padded(const Tensor& target, size_t nd, size_t idx) {
    size_t tnd = target.impl->ndim;
    if (idx < nd - tnd) return 1;
    return target.impl->shape[idx - (nd - tnd)];
}

// ------------------ accumulate_grad (broadcast-aware & PARALLEL) ------------------
inline void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) throw std::runtime_error("accumulate_grad: target undefined");
    if (!grad_src.impl) throw std::runtime_error("accumulate_grad: grad_src undefined");

    size_t nd_t = target.impl->ndim;
    size_t nd_g = grad_src.impl->ndim;

    std::vector<int> axes_to_reduce;
    size_t max_ndim = std::max(nd_t, nd_g);
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t td = dim_in_padded(target, max_ndim, i);
        size_t gd = dim_in_padded(grad_src, max_ndim, i);
        if (td == 1 && gd > 1) {
            axes_to_reduce.push_back((int)(i - (max_ndim - nd_g)));
        }
    }

    Tensor g_aligned = grad_src;
    if (!axes_to_reduce.empty()) {
        g_aligned = reduce_sum_axes_keepdims(grad_src, axes_to_reduce);
    }
    
    ensure_grad_buffer(target, false); // Do not zero existing grad!

    size_t N = target.numel();
    if (N == 0) return;

    bool g_has_gradbuf = (g_aligned.impl->storage->grad != nullptr);
    void* g_data_ptr = g_has_gradbuf ? g_aligned.impl->storage->grad.get() : g_aligned.impl->storage->data.get();
    void* t_grad_ptr = target.impl->storage->grad.get();

    // Pointers for OMP
    const size_t* t_shape = target.impl->shape;
    const size_t* t_strides = target.impl->strides;
    const size_t* g_strides = g_aligned.impl->strides;
    const size_t* g_shape = g_aligned.impl->shape;
    
    size_t t_offset = target.impl->offset;
    size_t g_offset = g_aligned.impl->offset;
    size_t t_ndim = target.impl->ndim;
    size_t g_ndim = g_aligned.impl->ndim;
    DType dt_g = g_aligned._dtype();
    DType dt_t = target._dtype();

    size_t pad = (t_ndim > g_ndim) ? (t_ndim - g_ndim) : 0;

    #pragma omp parallel for
    for (size_t flat = 0; flat < N; ++flat) {
        size_t rem = flat;
        size_t target_strided_idx = t_offset;
        size_t g_aligned_strided_idx = g_offset;

        // Decode index and compute offsets simultaneously
        for (int d = (int)t_ndim - 1; d >= 0; --d) {
            size_t sz = t_shape[d];
            size_t coord = rem % sz;
            rem /= sz;

            target_strided_idx += coord * t_strides[d];

            // Calculate corresponding g_aligned index (handling broadcast)
            if (d >= (int)pad) {
                size_t g_dim_idx = d - pad;
                // Broadcast check: if g_shape is 1, stride logic works naturally if we rely on logic
                // But explicit check for 1 is safer for views with 0 stride.
                if (g_shape[g_dim_idx] > 1) {
                    g_aligned_strided_idx += coord * g_strides[g_dim_idx];
                }
            }
        }

        double addv = read_scalar_at(g_data_ptr, g_aligned_strided_idx, dt_g);
        double cur = read_scalar_at(t_grad_ptr, target_strided_idx, dt_t);
        write_scalar_at(t_grad_ptr, target_strided_idx, dt_t, cur + addv);
    }
}


// ------------------ Backward nodes (Updated to use opsmp) ------------------

GradSub::GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradSub::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSub: missing self grad");
    if (a.requires_grad()) accumulate_grad(a, self);
    if (b.requires_grad()) {
        Tensor grad_self = tensor_from_grad(self);
        Tensor neg = grad_self * -1.0; // Uses opsmp operator*
        accumulate_grad(b, neg);
    }
}

GradMul::GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradMul::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMul: missing self grad");
    Tensor grad_self = tensor_from_grad(self); 

    if (a.requires_grad()) {
        Tensor ga = grad_self * b; // opsmp
        accumulate_grad(a, ga);
    }
    if (b.requires_grad()) {
        Tensor gb = grad_self * a; // opsmp
        accumulate_grad(b, gb);
    }
}

GradDiv::GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradDiv::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradDiv: missing self grad");
    Tensor grad_self = tensor_from_grad(self);

    bool old_grad_a = a.impl->requires_grad;
    bool old_grad_b = b.impl->requires_grad;
    a.impl->requires_grad = false;
    b.impl->requires_grad = false;

    if (old_grad_a) {
        Tensor da = grad_self / b; 
        accumulate_grad(a, da);
    }
    if (old_grad_b) {
        Tensor num = grad_self * a; 
        Tensor den = b * b;         
        Tensor db = (num / den) * -1.0; 
        accumulate_grad(b, db);
    }
    a.impl->requires_grad = old_grad_a;
    b.impl->requires_grad = old_grad_b;
}

GradMatMul::GradMatMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
void GradMatMul::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMatMul: missing self grad");

    Tensor grad_y = tensor_from_grad(self); 

    auto transpose_last_two = [](const Tensor &t) -> Tensor {
        if (t.impl->ndim < 2) return t.clone();
        std::vector<size_t> perm(t.impl->ndim);
        for (size_t i = 0; i < t.impl->ndim; ++i) perm[i] = i;
        std::swap(perm[t.impl->ndim - 2], perm[t.impl->ndim - 1]);
        return t.permute(perm);
    };

    if (a.requires_grad()) {
        Tensor bt = transpose_last_two(b);
        Tensor grad_a = matmul_mp(grad_y, bt); // Explicit call to MP
        accumulate_grad(a, grad_a);
    }
    if (b.requires_grad()) {
        Tensor at = transpose_last_two(a);
        Tensor grad_b = matmul_mp(at, grad_y); // Explicit call to MP
        accumulate_grad(b, grad_b);
    }
}

GradPow::GradPow(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradPow::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradPow: missing self grad");
    Tensor grad_y = tensor_from_grad(self);

    if (a.requires_grad()) {
        bool old_a = a.requires_grad();
        bool old_b = b.requires_grad();
        a.impl->requires_grad = false; b.impl->requires_grad = false;

        Tensor grad_a = grad_y * b * pow_mp(a, b - 1.0); // Explicit MP
        accumulate_grad(a, grad_a);
        
        a.impl->requires_grad = old_a; b.impl->requires_grad = old_b;
    }
    if (b.requires_grad()) {
        bool old_a = a.requires_grad();
        bool old_b = b.requires_grad();
        a.impl->requires_grad = false; b.impl->requires_grad = false;

        Tensor grad_b = grad_y * pow_mp(a, b) * ln_mp(a); // Explicit MP
        accumulate_grad(b, grad_b);

        a.impl->requires_grad = old_a; b.impl->requires_grad = old_b;
    }
}

//Scalar
void GradAddScalar::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradAddScalar: missing self grad");
    if (!a.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    accumulate_grad(a, grad_input);
}

void GradSubScalar::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSubScalar: missing self grad");
    if (!a.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    accumulate_grad(a, grad_input);
}

void GradSubAfterScalar::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSubAfterScalar: missing self grad");
    if (!a.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    grad_input = grad_input * -1.0;
    accumulate_grad(a, grad_input);
}
void GradMulScalar::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMulScalar: missing self grad");
    if (!a.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    grad_input = grad_input * scalar;
    accumulate_grad(a, grad_input);
}

void GradLn::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradLn: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_ln = grad_input / t; 
    t.impl->requires_grad = old;
    
    accumulate_grad(t, grad_ln);
}
void GradExp::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradExp: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);

    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_exp = grad_input * exp_mp(t);
    t.impl->requires_grad = old;
    
    accumulate_grad(t, grad_exp);
}
void GradSqrt::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSqrt: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_sqrt = grad_input / (sqrt_mp(t) * 2.0);
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_sqrt);
}

void GradSin::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSin: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_sin = grad_input * cos_mp(t);
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_sin);
}

void GradASin::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradASin: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);

    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = pow_scalar_mp(1.0 - (t * t), -0.5);
    Tensor grad_asin = grad_input * deriv;
    t.impl->requires_grad = old;
    
    accumulate_grad(t, grad_asin);
}
void GradSinH::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSinH: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_sin = grad_input * cosh_mp(t);
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_sin);
}

void GradCos::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradCos: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_cos = grad_input * sin_mp(t) * -1.0;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_cos);
}

void GradACos::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradACos: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = pow_scalar_mp(1.0 - (t * t), -0.5) * -1.0;
    Tensor grad_acos = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_acos);
}

void GradCosH::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradCosH: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor grad_cos = grad_input * sinh_mp(t); 
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_cos);
}

void GradTan::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradTan: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor cos_t = cos_mp(t);
    Tensor deriv = 1.0 / (cos_t * cos_t);
    Tensor grad_tan = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_tan);
}
void GradATan::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradATan: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = 1.0 / (1.0 + (t * t));
    Tensor grad_atan = grad_input * deriv;
    t.impl->requires_grad = old;
    
    accumulate_grad(t, grad_atan);
}
void GradTanH::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradTanH: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor tanh_t = tanh_mp(t);
    Tensor deriv = 1.0 - (tanh_t * tanh_t);
    Tensor grad_tanh = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_tanh);
}

void GradSigmoid::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("Sigmoid: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = self * (1.0 - self);
    Tensor grad_sig = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_sig);
}
void GradRelu::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("Relu: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = gt_mp(t, 0.0); 
    Tensor grad_relu = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_relu);
}
void GradSoftPlus::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("SoftPlus: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);

    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = sigmoid_mp(t);
    Tensor grad_sp = grad_input * deriv;
    t.impl->requires_grad = old;

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
    if (!self.impl->storage->grad) throw std::runtime_error("GradAbs: missing self grad");
    if (!t.requires_grad()) return;
    Tensor grad_input = tensor_from_grad(self);
    
    bool old = t.requires_grad();
    t.impl->requires_grad = false;
    Tensor deriv = gt_mp(t, 0.0) - lt_mp(t, 0.0);
    Tensor grad_abs = grad_input * deriv;
    t.impl->requires_grad = old;

    accumulate_grad(t, grad_abs);
}

void GradSum::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradSum: missing self grad");
    if (!t.requires_grad()) return;

    // Scalar reduction returns a 0-dim or 1-dim tensor at offset 0 usually.
    double g = read_scalar_at(self.impl->storage->grad.get(), self.impl->offset, self.impl->dtype);

    std::vector<size_t> shape_vec(t.impl->shape, t.impl->shape + t.impl->ndim);
    Tensor grad_input = Tensor::full(shape_vec, g, t.impl->dtype, false);

    accumulate_grad(t, grad_input);
}
void GradMean::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMean: missing self grad");
    if (!t.requires_grad()) return;

    // Scalar reduction returns a 0-dim or 1-dim tensor at offset 0 usually.
    double g = read_scalar_at(self.impl->storage->grad.get(), self.impl->offset, self.impl->dtype);
    g = g / static_cast<double>(t.numel());
    std::vector<size_t> shape_vec(t.impl->shape, t.impl->shape + t.impl->ndim);
    Tensor grad_input = Tensor::full(shape_vec, g, t.impl->dtype, false);

    accumulate_grad(t, grad_input);
}


// ------------------ backward ------------------
void backward(Tensor& loss) {
    if (!loss.impl) throw std::runtime_error("backward: loss undefined");
    if (!loss.impl->requires_grad) throw std::runtime_error("backward: loss requires_grad == false");

    ensure_grad_buffer(loss, true); // Initialize loss grad to zero first
    
    size_t n = loss.numel();
    std::vector<size_t> idx_vec(loss.impl->ndim, 0);
    
    // Fill with 1.0
    // Optimization: If loss is scalar, just write 1.0.
    // Using parallel loop for generality
    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t strided_idx = loss.impl->offset;
        // Local index reconstruction
        for (int d = (int)loss.impl->ndim - 1; d >= 0; --d) {
            size_t sz = loss.impl->shape[d];
            size_t coord = rem % sz;
            rem /= sz;
            strided_idx += coord * loss.impl->strides[d];
        }
        write_scalar_at(loss.impl->storage->grad.get(), strided_idx, loss._dtype(), 1.0);
    }

    std::vector<Tensor> topo;
    topo_sort_from(loss, topo);

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
