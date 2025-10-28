// ------------------- autograd core -------------------
#include <unordered_map>
#include <functional>
#include <stack>
#include <set>
#include "tensor1.h"

struct GradFn {
    std::string op;
    std::vector<Tensor> parents;
    // signature: (self_tensor, parents)
    std::function<void(const Tensor& self, const std::vector<Tensor>& parents)> backward;
};

// registry type
inline std::unordered_map<std::string,
    std::function<void(const Tensor& self, const std::vector<Tensor>& parents)>>
    grad_registry;

// register helper
inline void register_grad(const std::string& name,
    std::function<void(const Tensor& self, const std::vector<Tensor>& parents)> fn) {
    grad_registry[name] = std::move(fn);
}

// helper: ensure grad buffer exists (zeros)
inline void ensure_grad_alloc(Tensor &t) {
    if (!t.impl->requires_grad) return;
    if (!t.impl->storage->grad) {
        size_t n = t.numel_();
        void* g = std::calloc(n, dtype_size(t.impl->dtype));
        if (!g && n) throw std::bad_alloc();
        t.impl->storage->grad = std::shared_ptr<void>(g, std::free);
    }
}

// topological sort (DFS)
static void topo_sort_from(const Tensor& root, std::vector<Tensor>& topo) {
    std::set<const Tensorimpl*> visited;
    std::function<void(const Tensor&)> dfs = [&](const Tensor& t){
        auto impl_ptr = t.impl.get();
        if (!impl_ptr) return;
        if (visited.count(impl_ptr)) return;
        visited.insert(impl_ptr);
        if (t.impl->grad_fn) {
            for (auto &p : t.impl->grad_fn->parents) dfs(p);
        }
        topo.push_back(t);
    };
    dfs(root);
}

// ------------------- Tensor::backward (add to your Tensor) -------------------
inline void Tensor::backward() {
    if (!impl) throw std::runtime_error("backward on empty tensor");
    if (!impl->requires_grad) throw std::runtime_error("backward: requires_grad is false");

    // init grad of root to ones (shape aware)
    ensure_grad_alloc(*this);
    size_t n = numel_();
    for (size_t i = 0; i < n; ++i) write_scalar_at(impl->storage->grad.get(), i, impl->dtype, 1.0);

    // build topo order
    std::vector<Tensor> topo;
    topo_sort_from(*this, topo);

    // traverse in reverse topo (root last -> parents earlier)
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor cur = *it;
        if (!cur.impl->grad_fn) continue;

        // call the registered backward; the GradFn stored the function pointer too but we
        // use the GradFn.backward (already set at op creation) for flexibility.
        auto fn = cur.impl->grad_fn->backward;
        if (fn) {
            // ensure parent's grad buffers exist before accumulation inside op's backward
            for (auto &p : cur.impl->grad_fn->parents) ensure_grad_alloc(const_cast<Tensor&>(p));
            fn(cur, cur.impl->grad_fn->parents);
        }
    }
}
