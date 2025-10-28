// autograd_core.h / autograd_core.cpp (merge into your file as needed)
#include <unordered_map>
#include <functional>
#include <stack>
#include <set>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

#include "tensor1.h"   // your Tensor implementation
#include "ops1.h"      // forward ops: add_, sub_, mul_, div_, matmul_, pow_scalar_ etc.

// ------------------ accumulate_grad (fixed to read grad buffer) ------------------
inline void accumulate_grad(Tensor& target, const Tensor& grad_src) {
    if (!target.impl) throw std::runtime_error("accumulate_grad: target undefined");
    if (!grad_src.impl) throw std::runtime_error("accumulate_grad: grad_src undefined");

    // ensure grad buffer exists on target
    if (!target.impl->storage->grad) {
        size_t nbytes = target.numel() * target.dtype_bytes();
        void* gptr = std::malloc(nbytes);
        if (!gptr && nbytes) throw std::bad_alloc();
        std::memset(gptr, 0, nbytes);
        target.impl->storage->grad = std::shared_ptr<void>(gptr, std::free);
    }

    // grad_src must have its gradient stored in grad_src.impl->storage->grad
    if (!grad_src.impl->storage->grad)
        throw std::runtime_error("accumulate_grad: grad_src has no grad buffer");

    size_t n = target.numel();
    for (size_t i = 0; i < n; ++i) {
        double cur = read_scalar_at(target.impl->storage->grad.get(), i, target._dtype());
        double addv = read_scalar_at(grad_src.impl->storage->grad.get(), i, grad_src._dtype());
        write_scalar_at(target.impl->storage->grad.get(), i, target._dtype(), cur + addv);
    }
}

// ------------------ GradFn base ------------------
struct GradFn {
    std::vector<Tensor> parents;              // important: used for DFS/topo traversal
    virtual void backward(const Tensor& self) = 0; // self is the tensor whose grad is in self.impl->storage->grad
    virtual ~GradFn() = default;
};

// ------------------ Add / Sub / Mul / Div / Pow / Matmul backward nodes ------------------

// Add: d(a+b)/da = 1 * grad_self
struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {
        parents = { a, b };
    }
    void backward(const Tensor& self) override {
        if (a.requires_grad()) accumulate_grad(a, self);
        if (b.requires_grad()) accumulate_grad(b, self);
    }
};

// Sub: d(a-b)/da = 1, d(a-b)/db = -1
struct GradSub : GradFn {
    Tensor a, b;
    GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {
        parents = { a, b };
    }
    void backward(const Tensor& self) override {
        if (a.requires_grad()) accumulate_grad(a, self);
        if (b.requires_grad()) {
            // create negated grad buffer (in neg.impl->storage->grad)
            Tensor neg = self.clone(); // clone copies data buffer but not grad; we'll put gradient values into neg.impl->storage->grad
            size_t n = self.numel();
            void* gb = std::malloc(n * neg.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            neg.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i) {
                double v = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
                write_scalar_at(neg.impl->storage->grad.get(), i, neg._dtype(), -v);
            }
            accumulate_grad(b, neg);
        }
    }
};

// Mul: d(a*b)/da = b * grad_self ; d(...)/db = a * grad_self
struct GradMul : GradFn {
    Tensor a, b;
    GradMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {
        parents = { a, b };
    }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradMul: missing self grad");
        size_t n = self.numel();

        if (a.requires_grad()) {
            // prepare grad_self (in grad buffer)
            Tensor grad_self = self.clone();
            void* gb = std::malloc(n * grad_self.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            grad_self.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i) {
                double v = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
                write_scalar_at(grad_self.impl->storage->grad.get(), i, grad_self._dtype(), v);
            }
            // ga_data = grad_self * b (forward)
            Tensor ga_data = mul_(grad_self, b);
            // move ga_data.data -> ga_data.impl->storage->grad for accumulate_grad
            void* gb2 = std::malloc(n * ga_data.dtype_bytes());
            if (!gb2 && n) throw std::bad_alloc();
            ga_data.impl->storage->grad = std::shared_ptr<void>(gb2, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(ga_data.impl->storage->data.get(), i, ga_data._dtype());
                write_scalar_at(ga_data.impl->storage->grad.get(), i, ga_data._dtype(), vv);
            }
            accumulate_grad(a, ga_data);
        }

        if (b.requires_grad()) {
            Tensor grad_self = self.clone();
            void* gb = std::malloc(n * grad_self.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            grad_self.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i) {
                double v = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
                write_scalar_at(grad_self.impl->storage->grad.get(), i, grad_self._dtype(), v);
            }
            Tensor gb_data = mul_(grad_self, a);
            void* gb2 = std::malloc(n * gb_data.dtype_bytes());
            if (!gb2 && n) throw std::bad_alloc();
            gb_data.impl->storage->grad = std::shared_ptr<void>(gb2, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(gb_data.impl->storage->data.get(), i, gb_data._dtype());
                write_scalar_at(gb_data.impl->storage->grad.get(), i, gb_data._dtype(), vv);
            }
            accumulate_grad(b, gb_data);
        }
    }
};

// Div: z = a / b
// da = grad_self / b
// db = - grad_self * a / (b*b)
struct GradDiv : GradFn {
    Tensor a, b;
    GradDiv(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {
        parents = { a, b };
    }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradDiv: missing self grad");
        size_t n = self.numel();

        if (a.requires_grad()) {
            Tensor grad_self = self.clone();
            void* gb = std::malloc(n * grad_self.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            grad_self.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i)
                write_scalar_at(grad_self.impl->storage->grad.get(), i, grad_self._dtype(),
                                read_scalar_at(self.impl->storage->grad.get(), i, self._dtype()));
            Tensor da_data = div_(grad_self, b); // forward
            void* gb2 = std::malloc(n * da_data.dtype_bytes());
            if (!gb2 && n) throw std::bad_alloc();
            da_data.impl->storage->grad = std::shared_ptr<void>(gb2, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(da_data.impl->storage->data.get(), i, da_data._dtype());
                write_scalar_at(da_data.impl->storage->grad.get(), i, da_data._dtype(), vv);
            }
            accumulate_grad(a, da_data);
        }

        if (b.requires_grad()) {
            Tensor grad_self = self.clone();
            void* gb = std::malloc(n * grad_self.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            grad_self.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i)
                write_scalar_at(grad_self.impl->storage->grad.get(), i, grad_self._dtype(),
                                read_scalar_at(self.impl->storage->grad.get(), i, self._dtype()));

            Tensor num = mul_(grad_self, a);        // grad_self * a
            Tensor den = mul_(b, b);                // b*b
            Tensor db_data = div_(num, den);        // (grad*a)/(b*b)

            // negate db_data.data
            size_t m = db_data.numel();
            for (size_t i = 0; i < m; ++i) {
                double val = read_scalar_at(db_data.impl->storage->data.get(), i, db_data._dtype());
                write_scalar_at(db_data.impl->storage->data.get(), i, db_data._dtype(), -val);
            }

            // move data -> grad buffer for accumulate_grad
            void* gb2 = std::malloc(m * db_data.dtype_bytes());
            if (!gb2 && m) throw std::bad_alloc();
            db_data.impl->storage->grad = std::shared_ptr<void>(gb2, std::free);
            for (size_t i = 0; i < m; ++i) {
                double vv = read_scalar_at(db_data.impl->storage->data.get(), i, db_data._dtype());
                write_scalar_at(db_data.impl->storage->grad.get(), i, db_data._dtype(), vv);
            }
            accumulate_grad(b, db_data);
        }
    }
};

// Pow (scalar exponent): z = a ^ p ; da = p * a^(p-1) * grad_self
struct GradPowScalar : GradFn {
    Tensor a;
    double p;
    GradPowScalar(const Tensor& a_, double p_) : a(a_), p(p_) {
        parents = { a };
    }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradPowScalar: missing self grad");
        size_t n = self.numel();
        if (a.requires_grad()) {
            Tensor a_pow = pow_scalar_(a, p - 1.0); // forward
            for (size_t i = 0; i < n; ++i) {
                double v = read_scalar_at(a_pow.impl->storage->data.get(), i, a_pow._dtype());
                write_scalar_at(a_pow.impl->storage->data.get(), i, a_pow._dtype(), v * p);
            }
            Tensor grad_self = self.clone();
            void* gb = std::malloc(n * grad_self.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            grad_self.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i)
                write_scalar_at(grad_self.impl->storage->grad.get(), i, grad_self._dtype(),
                                read_scalar_at(self.impl->storage->grad.get(), i, self._dtype()));
            Tensor ga_data = mul_(grad_self, a_pow);
            void* gb2 = std::malloc(n * ga_data.dtype_bytes());
            if (!gb2 && n) throw std::bad_alloc();
            ga_data.impl->storage->grad = std::shared_ptr<void>(gb2, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(ga_data.impl->storage->data.get(), i, ga_data._dtype());
                write_scalar_at(ga_data.impl->storage->grad.get(), i, ga_data._dtype(), vv);
            }
            accumulate_grad(a, ga_data);
        }
    }
};

// MatMul (2D): z = a @ b
// da = grad_out @ b^T ; db = a^T @ grad_out
struct GradMatMul : GradFn {
    Tensor a, b;
    GradMatMul(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {
        parents = { a, b };
    }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradMatMul: missing self grad");

        if (a.requires_grad()) {
            Tensor bt = b.transpose(); // adapt to your API
            Tensor da = matmul_(self, bt);
            size_t n = da.numel();
            void* gb = std::malloc(n * da.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            da.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(da.impl->storage->data.get(), i, da._dtype());
                write_scalar_at(da.impl->storage->grad.get(), i, da._dtype(), vv);
            }
            accumulate_grad(a, da);
        }
        if (b.requires_grad()) {
            Tensor at = a.transpose();
            Tensor db = matmul_(at, self);
            size_t n = db.numel();
            void* gb = std::malloc(n * db.dtype_bytes());
            if (!gb && n) throw std::bad_alloc();
            db.impl->storage->grad = std::shared_ptr<void>(gb, std::free);
            for (size_t i = 0; i < n; ++i) {
                double vv = read_scalar_at(db.impl->storage->data.get(), i, db._dtype());
                write_scalar_at(db.impl->storage->grad.get(), i, db._dtype(), vv);
            }
            accumulate_grad(b, db);
        }
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
            // follow parents stored in grad_fn
            for (const Tensor& p : t.impl->grad_fn->parents) {
                dfs(p);
            }
        }
        topo.push_back(t);
    };
    dfs(root);
}

// ------------------ backward (topo + reverse traversal) ------------------
void backward(Tensor& loss) {
    if (!loss.impl) throw std::runtime_error("backward: loss undefined");
    if (!loss.impl->requires_grad) throw std::runtime_error("backward: loss requires_grad == false");

    // ensure loss has a grad buffer and set to ones
    if (!loss.impl->storage->grad) {
        size_t nbytes = loss.numel() * loss.dtype_bytes();
        void* gptr = std::malloc(nbytes);
        if (!gptr && nbytes) throw std::bad_alloc();
        // fill with ones as double values
        if (loss._dtype() == DType::Float32 || loss._dtype() == DType::Double64) {
            // assume double-backed in your implementation; adjust if using float storage
            for (size_t i = 0; i < loss.numel(); ++i)
                write_scalar_at(gptr, i, loss._dtype(), 1.0);
        } else {
            std::memset(gptr, 0, nbytes);
        }
        loss.impl->storage->grad = std::shared_ptr<void>(gptr, std::free);
    } else {
        for (size_t i = 0; i < loss.numel(); ++i)
            write_scalar_at(loss.impl->storage->grad.get(), i, loss._dtype(), 1.0);
    }

    // build topo order
    std::vector<Tensor> topo;
    topo_sort_from(loss, topo);

    // traverse in reverse topo order (from loss toward leaves)
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor cur = *it;
        if (!cur.impl->grad_fn) continue;
        // ensure parent's grad buffers exist? Not needed here — children will call accumulate_grad
        cur.impl->grad_fn->backward(cur);
    }
}
