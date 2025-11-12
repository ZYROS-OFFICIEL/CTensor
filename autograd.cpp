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
#include "ops1.h"

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
// ------------------------------------------------------------
Tensor tensor_from_grad(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("tensor_from_grad: missing grad buffer");

    Tensor grad_tensor(self.shape(), self._dtype(), false);
    size_t n = self.numel_();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
        write_scalar_at(grad_tensor.impl->storage->data.get(), i, grad_tensor._dtype(), gv);
    }

    return grad_tensor;
}

// copy .data -> .grad (allocate grad buffer and copy values)
// after calling this the gradient values are available in impl->storage->grad
static void copy_data_to_grad(Tensor &t) {
    if (!t.impl) throw std::runtime_error("copy_data_to_grad: undefined tensor");
    size_t n = t.numel();
    ensure_grad_buffer(t, false);
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
        write_scalar_at(t.impl->storage->grad.get(), i, t.impl->dtype, v);
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

// Broadcast-aware accumulate_grad: map values from grad_src into target.impl->storage->grad
inline void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) throw std::runtime_error("accumulate_grad: target undefined");
    if (!grad_src.impl) throw std::runtime_error("accumulate_grad: grad_src undefined");

    // Ensure target has grad buffer allocated
    ensure_grad_buffer(target, true);

    // Step 1: determine axes that must be reduced (where target has dim=1 but grad_src has >1)
    size_t nd_t = target.impl->ndim;
    size_t nd_g0 = grad_src.impl->ndim;

    std::vector<int> axes_to_reduce;
    for (size_t i = 0; i < nd_g0; ++i) {
        size_t td = dim_in_padded(target, nd_g0, i);            // target dim when left-padded to nd_g0
        size_t gd = grad_src.impl->shape[i];
        if (td == 1 && gd > 1) axes_to_reduce.push_back((int)i);
    }

    // Step 2: reduce grad_src over axes_to_reduce (keep dims) so broadcasting is explicit
    Tensor g_aligned = grad_src;
    if (!axes_to_reduce.empty()) {
        g_aligned = reduce_sum_axes_keepdims(grad_src, axes_to_reduce);
    }

    // Step 3: If g_aligned has fewer dims than target, left-pad g_aligned so dims match
    // (conceptually we want to iterate over g_aligned with left-padding aligned to target)
    size_t nd_g = g_aligned.impl->ndim;
    if (nd_g < nd_t) {
        // pad grad to target ndim (left-pad with ones)
        g_aligned = pad_to_ndim(g_aligned, nd_t);
        nd_g = g_aligned.impl->ndim;
    }

    // At this point nd_g >= nd_t OR nd_g == nd_t, but mapping below handles nd_g > nd_t by left-padding target when reading.
    // We'll iterate over all elements of g_aligned and add them into target.grad using broadcasting mapping.

    // Precompute some shortcuts
    const size_t N = g_aligned.numel_();
    const auto *g_data_ptr  = g_aligned.impl->storage->data.get();
    const auto *g_grad_ptr  = g_aligned.impl->storage->grad.get();
    bool g_has_gradbuf = (g_grad_ptr != nullptr);

    // helper vectors for multi-index calculation
    std::vector<size_t> idx_g(nd_g, 0);

    for (size_t flat = 0; flat < N; ++flat) {
        // compute multi-index of g_aligned (from flat)
        size_t rem = flat;
        for (int d = (int)nd_g - 1; d >= 0; --d) {
            idx_g[d] = rem % g_aligned.impl->shape[d];
            rem /= g_aligned.impl->shape[d];
        }

        // compute flat index into g_aligned storage (respecting its offset & strides)
        size_t g_flat_index = g_aligned.impl->offset;
        for (size_t d = 0; d < nd_g; ++d) {
            g_flat_index += idx_g[d] * g_aligned.impl->strides[d];
        }

        // read value from grad buffer if present, otherwise from data
        double addv = g_has_gradbuf
            ? read_scalar_at(g_grad_ptr, g_flat_index, g_aligned._dtype())
            : read_scalar_at(g_data_ptr, g_flat_index, g_aligned._dtype());

        // map this g_aligned multi-index to the corresponding target index
        // If g_aligned.ndim > target.ndim the mapping is "left-padded" (i.e., skip first pad dims)
        size_t pad = (nd_g > nd_t) ? (nd_g - nd_t) : 0;
        size_t target_flat_idx = target.impl->offset;
        for (size_t d = 0; d < nd_g; ++d) {
            // target dimension size for this g dimension (1 if it was left-padded)
            size_t tdim = (d < pad) ? 1 : target.impl->shape[d - pad];
            size_t use_idx = (tdim == 1) ? 0 : idx_g[d];
            if (d >= pad) {
                target_flat_idx += use_idx * target.impl->strides[d - pad];
            }
        }

        // finally read current grad at target location and accumulate
        double cur = read_scalar_at(target.impl->storage->grad.get(), target_flat_idx, target._dtype());
        write_scalar_at(target.impl->storage->grad.get(), target_flat_idx, target._dtype(), cur + addv);
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
        // negated grad: create neg = -self.data (we can use diff_ with zero tensor)
        Tensor neg = self.clone();
        size_t n = neg.numel();
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            write_scalar_at(neg.impl->storage->data.get(), i, neg._dtype(), -v);
        }
        // copy data to grad buffer for accumulate
        copy_data_to_grad(neg);
        accumulate_grad(b, neg);
    }
}


// Mul

GradMul::GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradMul::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradMul: missing self grad");

    size_t n = self.numel();

    // Create grad_self whose DATA holds incoming gradient values
    Tensor grad_self= tensor_from_grad(self);  

    // ga = grad_self * b
    if (a.requires_grad()) {
        Tensor ga = mult_(grad_self, b);
        copy_data_to_grad(ga);   // copy ga.data → ga.grad
        accumulate_grad(a, ga);
    }

    // gb = grad_self * a
    if (b.requires_grad()) {
        Tensor gb = mult_(grad_self, a);
        copy_data_to_grad(gb);
        accumulate_grad(b, gb);
    }
}


// Div
GradDiv::GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradDiv::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradDiv: missing self grad");

    Tensor grad_self= tensor_from_grad(self);  

    // ---- In-place “detach” ----
    bool old_grad_a = a.impl->requires_grad;
    bool old_grad_b = b.impl->requires_grad;

    a.impl->requires_grad = false;
    b.impl->requires_grad = false;

    // ---- Compute gradients ----
    if (old_grad_a) {
        Tensor da = div_(grad_self, b);  // grad_self / b
        copy_data_to_grad(da);
        accumulate_grad(a, da);
    }
    if (old_grad_b) {
        Tensor num = mult_(grad_self, a);   // grad_self * a
        Tensor den = mult_(b, b);           // b * b
        Tensor db = div_(num, den);         // (grad * a) / (b * b)

        // negate db
        size_t m = db.numel();
        for (size_t i = 0; i < m; ++i) {
            double v = read_scalar_at(db.impl->storage->data.get(), i, db._dtype());
            write_scalar_at(db.impl->storage->data.get(), i, db._dtype(), -v);
        }

        copy_data_to_grad(db);
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

    // --- Step 1: Extract gradient wrt output (dL/dY) ---
    Tensor grad_y = tensor_from_grad(self); // creates a tensor with grad data in .data

    // --- Step 2: Define a small helper to transpose the last two dims ---
    auto transpose_last_two = [](const Tensor &t) -> Tensor {
        if (!t.impl)
            throw std::runtime_error("transpose_last_two: undefined tensor");
        if (t.impl->ndim < 2)
            return t.clone(); // nothing to transpose

        std::vector<size_t> perm(t.impl->ndim);
        for (size_t i = 0; i < t.impl->ndim; ++i)
            perm[i] = i;
        std::swap(perm[t.impl->ndim - 2], perm[t.impl->ndim - 1]);
        return t.permute(perm);
    };

    // --- Step 3: Compute grad w.r.t A ---
    if (a.requires_grad()) {
        Tensor bt = transpose_last_two(b);
        Tensor grad_a = matmul_(grad_y, bt);   // ∂L/∂A = ∂L/∂Y @ B^T
        copy_data_to_grad(grad_a);
        accumulate_grad(a, grad_a);
    }

    // --- Step 4: Compute grad w.r.t B ---
    if (b.requires_grad()) {
        Tensor at = transpose_last_two(a);
        Tensor grad_b = matmul_(at, grad_y);   // ∂L/∂B = A^T @ ∂L/∂Y
        copy_data_to_grad(grad_b);
        accumulate_grad(b, grad_b);
    }
}



// Pow elementwise: z = a^b (both tensors)
// da = b * a^(b-1) * grad_self
// db = a^b * ln(a) * grad_self
GradPow::GradPow(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
void GradPow::backward(const Tensor& self) {
    if (!self.impl->storage->grad)
        throw std::runtime_error("GradPow: missing self grad");

    size_t n = self.numel();

    // 1️⃣ ∂Loss/∂y (grad from next layer)
    Tensor grad_y(self.shape(), self._dtype(), false);
    for (size_t i = 0; i < n; ++i) {
        double gy = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
        write_scalar_at(grad_y.impl->storage->data.get(), i, grad_y._dtype(), gy);
    }

    // 2️⃣ ∂Loss/∂a = grad_y * b * a^(b-1)
    if (a.requires_grad()) {
        Tensor grad_a(self.shape(), self._dtype(), false);
        for (size_t i = 0; i < n; ++i) {
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            double vb = read_scalar_at(b.impl->storage->data.get(), i, b._dtype());
            double gy = read_scalar_at(grad_y.impl->storage->data.get(), i, grad_y._dtype());
            double da = gy * vb * std::pow(va, vb - 1.0);
            write_scalar_at(grad_a.impl->storage->data.get(), i, grad_a._dtype(), da);
        }
        copy_data_to_grad(grad_a);
        accumulate_grad(a, grad_a);
    }

    // 3️⃣ ∂Loss/∂b = grad_y * a^b * ln(a)
    if (b.requires_grad()) {
        Tensor grad_b(self.shape(), self._dtype(), false);
        for (size_t i = 0; i < n; ++i) {
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            double vb = read_scalar_at(b.impl->storage->data.get(), i, b._dtype());
            double gy = read_scalar_at(grad_y.impl->storage->data.get(), i, grad_y._dtype());
            double safe_va = (va <= 0.0) ? 1e-12 : va;
            double db = gy * std::pow(va, vb) * std::log(safe_va);
            write_scalar_at(grad_b.impl->storage->data.get(), i, grad_b._dtype(), db);
        }
        copy_data_to_grad(grad_b);
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
    grad_input = grad_input * (-1);
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
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv / tv);
    }

    accumulate_grad(t, grad_input);
}
void GradExp::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradExp: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * std::exp(tv));
    }

    accumulate_grad(t, grad_input);
}
void GradSqrt::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSqrt: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * 1.0 / (2.0 * std::sqrt(tv)));
    }

    accumulate_grad(t, grad_input);
}
void GradSin::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSin: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * std::cos(tv));
    }

    accumulate_grad(t, grad_input);
}

void GradASin::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradASin: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * (1.0 / std::sqrt(1.0 - tv * tv)));
    }

    accumulate_grad(t, grad_input);
}

void GradCos::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradCos: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * (-std::sin(tv)));
    }

    accumulate_grad(t, grad_input);
}
void GradACos::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradACos: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * (-1.0 / std::sqrt(1.0 - tv * tv)));
    }

    accumulate_grad(t, grad_input);
}
void GradCosH::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradCoshH: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * std::sinh(tv));
    }

    accumulate_grad(t, grad_input);
}
void GradTan::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradTan: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(),gv * (1.0 / (std::cos(tv) * std::cos(tv))));
    }

    accumulate_grad(t, grad_input);
}
void GradATan::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradATan: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(),gv * (1.0 / (1.0 + tv * tv)));
    }

    accumulate_grad(t, grad_input);
}
void GradTanH::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradTanH: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * (1.0 - std::tanh(tv) * std::tanh(tv)));
    }

    accumulate_grad(t, grad_input);
}

void GradSigmoid::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("Sigmoid: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * tv * (1.0 - tv));
    }

    accumulate_grad(t, grad_input);
}
void GradRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("Relu: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), (tv > 0.0) ? gv : 0.0);
    }

    accumulate_grad(t, grad_input);
}
void GradSoftPlus::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("SoftPlus: missing self grad");
    if (!t.impl || !t.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        write_scalar_at(g_data, i, grad_input._dtype(), gv * (1.0 / (1.0 + std::exp(-tv))));
    }

    accumulate_grad(t, grad_input);
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
    size_t n = t.numel_();

    auto* g_data = grad_input.impl->storage->data.get();
    auto* t_data = t.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double gv = read_scalar_at(g_data, i, grad_input._dtype());
        double tv = read_scalar_at(t_data, i, t._dtype());
        double sign = (tv > 0.0) ? 1.0 : ((tv < 0.0) ? -1.0 : 0.0);
        write_scalar_at(g_data, i, grad_input._dtype(), gv * sign);
    }

    accumulate_grad(t, grad_input);
}

void GradSum::backward(const Tensor& self) {
    // self.impl->storage->grad must exist (scalar gradient)
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradSum: missing self grad");

    if (!t.impl || !t.requires_grad()) return;

    // scalar gradient value (assume scalar stored at index 0)
    double g = read_scalar_at(self.impl->storage->grad.get(), 0, self.impl->dtype);

    // create grad tensor of same shape as t and fill with g
    std::vector<size_t> shape_vec(t.impl->shape, t.impl->shape + t.impl->ndim);
    Tensor grad_input(shape_vec, t.impl->dtype, false); // grad itself does not require grad

    size_t n = t.numel_();
    for (size_t i = 0; i < n; ++i)
        write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input.impl->dtype, g);

    // accumulate into t's grad storage via existing helper
    accumulate_grad(t, grad_input);
}


// ------------------ backward ------------------
void backward(Tensor& loss) {
    if (!loss.impl) throw std::runtime_error("backward: loss undefined");
    if (!loss.impl->requires_grad) throw std::runtime_error("backward: loss requires_grad == false");

    // set loss grad to ones
    ensure_grad_buffer(loss, true);
    for (size_t i = 0; i < loss.numel(); ++i)
        write_scalar_at(loss.impl->storage->grad.get(), i, loss._dtype(), 1.0);

    // build topo order and run reverse
    std::vector<Tensor> topo;
    topo_sort_from(loss, topo);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor cur = *it;
        if (!cur.impl->grad_fn) continue;
        cur.impl->grad_fn->backward(cur);
    }
}
