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

#include "tensor1.h"
#include "ops1.h"

// -------------------- helpers --------------------

// ensure grad buffer exists on tensor; if zero=true fill with zeros
inline void ensure_grad_buffer(Tensor &t, bool zero = true) {
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

// copy .data -> .grad (allocate grad buffer and copy values)
// after calling this the gradient values are available in impl->storage->grad
inline void copy_data_to_grad(Tensor &t) {
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

// ------------------ GradFn base ------------------
struct GradFn {
    std::vector<Tensor> parents;              // used for DFS/topo traversal
    virtual void backward(const Tensor& self) = 0; // self is the tensor whose grad is in self.impl->storage->grad
    virtual ~GradFn() = default;
};

// ------------------ Backward nodes (use ops1 names) ------------------

// Add
struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
        // self.impl->storage->grad must exist
        if (!self.impl->storage->grad) throw std::runtime_error("GradAdd: missing self grad");
        if (a.requires_grad()) accumulate_grad(a, self);
        if (b.requires_grad()) accumulate_grad(b, self);
    }
};

// Sub (diff_)
struct GradSub : GradFn {
    Tensor a, b;
    GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
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
};

// Mul
struct GradMul : GradFn {
    Tensor a, b;
    GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradMul: missing self grad");
        size_t n = self.numel();

        // create grad_self as a Tensor whose .data is irrelevant but .grad has the gradients from self
        Tensor grad_self = self.clone();
        copy_data_to_grad(grad_self); // copy self.data or grad into grad buffer? Here self.impl->storage->grad holds backprop grad, so copy it:
        // But copy_data_to_grad copies data -> grad. For safety, we instead ensure grad_self.impl->storage->grad equals self.impl->storage->grad
        // (cheap pointer aliasing is OK since read-only). We'll alias to avoid copy:
        grad_self.impl->storage->grad = self.impl->storage->grad;

        if (a.requires_grad()) {
            // ga = grad_self * b (forward elementwise)
            Tensor ga = mult_(grad_self, b);
            // ga.data contains the gradient values; move to grad buffer
            copy_data_to_grad(ga);
            accumulate_grad(a, ga);
        }
        if (b.requires_grad()) {
            Tensor gb = mult_(grad_self, a);
            copy_data_to_grad(gb);
            accumulate_grad(b, gb);
        }
    }
};

// Div
struct GradDiv : GradFn {
    Tensor a, b;
    GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradDiv: missing self grad");
        Tensor grad_self = self.clone();
        // alias grad buffer
        grad_self.impl->storage->grad = self.impl->storage->grad;

        if (a.requires_grad()) {
            Tensor da = div_(grad_self, b); // (grad_self) / b
            copy_data_to_grad(da);
            accumulate_grad(a, da);
        }
        if (b.requires_grad()) {
            // db = - grad_self * a / (b*b)
            Tensor num = mult_(grad_self, a);   // grad_self * a
            Tensor den = mult_(b, b);           // b*b
            Tensor db = div_(num, den);         // (grad*a)/(b*b)
            // negate data in db.data
            size_t m = db.numel();
            for (size_t i = 0; i < m; ++i) {
                double v = read_scalar_at(db.impl->storage->data.get(), i, db._dtype());
                write_scalar_at(db.impl->storage->data.get(), i, db._dtype(), -v);
            }
            copy_data_to_grad(db);
            accumulate_grad(b, db);
        }
    }
};

// Pow elementwise: z = a^b (both tensors)
// da = b * a^(b-1) * grad_self
// db = a^b * ln(a) * grad_self
struct GradPow : GradFn {
    Tensor a, b;
    GradPow(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradPow: missing self grad");
        // alias grad buffer
        Tensor grad_self = self.clone();
        grad_self.impl->storage->grad = self.impl->storage->grad;
        size_t n = self.numel();

        if (a.requires_grad()) {
            // a_pow = a^(b-1)
            // compute b-1 as a data tensor: b_minus = b.data - 1
            Tensor b_minus = b.clone();
            for (size_t i = 0; i < n; ++i) {
                double vb = read_scalar_at(b.impl->storage->data.get(), i, b._dtype());
                write_scalar_at(b_minus.impl->storage->data.get(), i, b_minus._dtype(), vb - 1.0);
            }
            Tensor a_pow = pow_(a, b_minus); // pow_(a, b-1)
            // multiply by b: tmp = b * a_pow
            Tensor tmp = mult_(b, a_pow);
            // ga = grad_self * tmp
            Tensor ga = mult_(grad_self, tmp);
            copy_data_to_grad(ga);
            accumulate_grad(a, ga);
        }

        if (b.requires_grad()) {
            // db = a^b * ln(a) * grad_self
            // compute ln(a) elementwise into log_a (requires a > 0 for real ln)
            Tensor log_a = a.clone();
            for (size_t i = 0; i < n; ++i) {
                double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
                // numerical safety: if va <= 0 replace with small positive to avoid nan/infs
                double safe_va = (va <= 0.0) ? 1e-12 : va;
                write_scalar_at(log_a.impl->storage->data.get(), i, log_a._dtype(), std::log(safe_va));
            }
            // a_pow_b = self.data (already forward a^b is stored in self.impl->storage->data)
            Tensor a_pow_b = self.clone(); // contains data of a^b
            // compute tmp = a_pow_b * log_a
            Tensor tmp = mult_(a_pow_b, log_a);
            // db = tmp * grad_self
            Tensor db = mult_(tmp, grad_self);
            copy_data_to_grad(db);
            accumulate_grad(b, db);
        }
    }
};

// MatMul
struct GradMatMul : GradFn {
    Tensor a, b;
    GradMatMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a,b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradMatMul: missing self grad");
        // alias grad buffer
        Tensor grad_self = self.clone();
        grad_self.impl->storage->grad = self.impl->storage->grad;

        // transpose last two dims of b and a using permute
        auto transpose_last_two = [](const Tensor &t)->Tensor {
            if (!t.impl) throw std::runtime_error("transpose_last_two: undefined tensor");
            if (t.impl->ndim < 2) return t.clone();
            std::vector<size_t> perm(t.impl->ndim);
            for (size_t i = 0; i < t.impl->ndim; ++i) perm[i] = i;
            std::swap(perm[t.impl->ndim - 2], perm[t.impl->ndim - 1]);
            return t.permute(perm);
        };

        if (a.requires_grad()) {
            Tensor bt = transpose_last_two(b);
            Tensor da = matmul_(grad_self, bt);
            copy_data_to_grad(da);
            accumulate_grad(a, da);
        }
        if (b.requires_grad()) {
            Tensor at = transpose_last_two(a);
            Tensor db = matmul_(at, grad_self);
            copy_data_to_grad(db);
            accumulate_grad(b, db);
        }
    }
};
struct GradSum : GradFn {
    Tensor t;
    int dim;
    GradSum(const Tensor& t_, int dim_) : t(t_), dim(dim_) { parents = {t}; }

    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad)
            throw std::runtime_error("GradSum: missing self grad");

        // broadcast gradient back to shape of t
        Tensor grad_input(t.impl->shape, t.impl->dtype, false);
        size_t n = t.numel_();
        double g = read_scalar_at(self.impl->storage->grad.get(), 0, t.impl->dtype);
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(grad_input.impl->storage->data.get(), i, t.impl->dtype, g);
        accumulate_grad(t, grad_input);
    }
};


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
