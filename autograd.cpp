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
static Tensor tensor_from_grad(const Tensor& self) {
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

// ------------------ accumulate_grad (broadcast-aware) ------------------
// accumulates gradient from grad_src into target.
// grad_src may have its values in .impl->storage->grad (preferred) or in .impl->storage->data
inline void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) throw std::runtime_error("accumulate_grad: target undefined");
    if (!grad_src.impl) throw std::runtime_error("accumulate_grad: grad_src undefined");

    size_t nd_t = target.impl->ndim;
    size_t nd_g = grad_src.impl->ndim;

    // If target.ndim < grad_src.ndim, we treat target as left-padded with 1s.
    // Determine axes where target has 1 and grad has >1 (these axes need reduction).
    std::vector<int> axes_to_reduce;
    for (size_t i = 0; i < nd_g; ++i) {
        size_t td = dim_in_padded(target, nd_g, i);
        size_t gd = grad_src.impl->shape[i];
        if (td == 1 && gd > 1) axes_to_reduce.push_back((int)i);
    }

    // If no axes to reduce AND shapes match elementwise (after padding) we can accumulate directly
    bool shapes_match = true;
    if (nd_t != nd_g) shapes_match = false;
    else {
        for (size_t i = 0; i < nd_g; ++i) {
            size_t td = dim_in_padded(target, nd_g, i);
            size_t gd = grad_src.impl->shape[i];
            if (td != gd) { shapes_match = false; break; }
        }
    }

    // Prepare a grad tensor `g_aligned` that has same shape as `target` (possibly with singleton dims)
    Tensor g_aligned = grad_src;
    if (!axes_to_reduce.empty()) {
        // reduce sum over axes_to_reduce but keep dims (so shape becomes same as grad_src but with ones on reduced axes)
        g_aligned = reduce_sum_axes_keepdims(grad_src, axes_to_reduce);
    }
    // Now g_aligned has same ndim as grad_src; if target.ndim < g_aligned.ndim, we must squeeze padding:
    // We want a tensor with exactly the same layout/ndim as target (left-padded). If target.ndim < g_aligned.ndim,
    // we need to remove leftmost leading dims that correspond to padding in target.
    if (target.impl->ndim < g_aligned.impl->ndim) {
        // remove leftmost dims where target is implicitly size 1 (i.e., drop dims until nd matches)
        size_t drop = g_aligned.impl->ndim - target.impl->ndim;
        // we can sum/reshape, but easiest is to create a view via reshape to match target.numel if contiguous.
        // For simplicity we will *reshape* by using `reshape` only if total elements match.
        // Convert g_aligned to contiguous flat tensor and then reshape to target.numel
        // But we can instead create a new Tensor g2 with shape padded to target.impl->ndim and copy values elementwise.
        // Simpler approach: use pad_to_ndim on target to expand it to g_aligned.ndim, then we will compare shapes.
        Tensor t_padded = pad_to_ndim(target, g_aligned.impl->ndim);
        // now t_padded and g_aligned have same ndim; we only need to ensure their shapes match (they should)
        // We'll proceed to elementwise accumulation using t_padded as mapping reference, but accumulation target must be original target.
        // To do this, we'll create a temp container (vector) for g_aligned values then map into target indices.
        // For simplicity and correctness (not optimal performance) we will iterate over all elements of g_aligned (flat)
        // and add mapped values to target grad using broadcasting index mapping logic.
        size_t N = g_aligned.numel_();
        // ensure target grad buffer
        ensure_grad_buffer(target, true);
        std::vector<size_t> idx(g_aligned.impl->ndim, 0);
        for (size_t flat = 0; flat < N; ++flat) {
            size_t rem = flat;
            for (int d = (int)g_aligned.impl->ndim - 1; d >= 0; --d) {
                idx[d] = rem % g_aligned.impl->shape[d];
                rem /= g_aligned.impl->shape[d];
            }
            // compute flat index in target (taking into account broadcasting in target)
            size_t target_flat = 0;
            size_t pad = g_aligned.impl->ndim - target.impl->ndim;
            for (size_t d = 0; d < g_aligned.impl->ndim; ++d) {
                size_t tdim = (d < pad) ? 1 : target.impl->shape[d - pad];
                size_t use_idx = (tdim == 1) ? 0 : idx[d];
                if (d >= pad) target_flat += use_idx * target.impl->strides[d - pad];
            }
            // read grad aligned value (prefer grad buffer, else data)
            double addv;
            if (g_aligned.impl->storage->grad)
                addv = read_scalar_at(g_aligned.impl->storage->grad.get(), flat, g_aligned._dtype());
            else
                addv = read_scalar_at(g_aligned.impl->storage->data.get(), flat, g_aligned._dtype());

            double cur = read_scalar_at(target.impl->storage->grad.get(), target_flat, target._dtype());
            write_scalar_at(target.impl->storage->grad.get(), target_flat, target._dtype(), cur + addv);
        }
        return;
    }

    // If we reach here, either shapes matched elementwise (after possible keepdims) or nd_t==nd_g
    // Ensure target has grad buffer
    ensure_grad_buffer(target, true);

    // Now simply add elementwise. We'll map each flat index of target to corresponding index in g_aligned:
    size_t N = target.numel_();
    // If g_aligned has grad buffer, read from it; else read from data
    bool g_has_gradbuf = (g_aligned.impl->storage->grad != nullptr);

    // g_aligned and target now must have same ndim; we still need to handle cases where g_aligned has shape=1 along some dims and target >1 (but that would have been axes_to_reduce earlier)
    // We'll compute flat-wise mapping with broadcast check
    std::vector<size_t> idx_vec(target.impl->ndim, 0);
    for (size_t flat = 0; flat < N; ++flat) {
        // compute multi-index for target
        size_t rem = flat;
        for (int d = (int)target.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % target.impl->shape[d];
            rem /= target.impl->shape[d];
        }
        // compute corresponding index in g_aligned
        size_t idx_g = 0;
        for (size_t d = 0; d < target.impl->ndim; ++d) {
            size_t gd = g_aligned.impl->shape[d];
            size_t use = (gd == 1 ? 0 : idx_vec[d]);
            idx_g += use * g_aligned.impl->strides[d];
        }
        double addv = g_has_gradbuf ? read_scalar_at(g_aligned.impl->storage->grad.get(), idx_g, g_aligned._dtype())
                                     : read_scalar_at(g_aligned.impl->storage->data.get(), idx_g, g_aligned._dtype());
        double cur = read_scalar_at(target.impl->storage->grad.get(), flat, target._dtype());
        write_scalar_at(target.impl->storage->grad.get(), flat, target._dtype(), cur + addv);
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

void GradMulScalar::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMulScalar: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    grad_input *= scalar;  // reuse existing op
    accumulate_grad(a, grad_input);
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
