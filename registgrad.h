// ------------------- register add -------------------
register_grad("add", [](const Tensor& self, const std::vector<Tensor>& parents){
    // parents[0] = a, parents[1] = b
    const Tensor &a = parents[0];
    const Tensor &b = parents[1];
    // self.impl->storage->grad contains dL/dself
    if (!self.impl->storage->grad) throw std::runtime_error("add backward: missing grad for self");

    size_t n = self.numel_();

    // distribute grad to a (respect broadcasting)
    if (a.impl->requires_grad) {
        // allocate if missing (should be ensured by caller, but safe-guard)
        ensure_grad_alloc(const_cast<Tensor&>(a));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double g_a   = read_scalar_at(a.impl->storage->grad.get(), i, a._dtype());
            write_scalar_at(a.impl->storage->grad.get(), i, a._dtype(), g_a + g_self);
        }
    }
    if (b.impl->requires_grad) {
        ensure_grad_alloc(const_cast<Tensor&>(b));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double g_b   = read_scalar_at(b.impl->storage->grad.get(), i, b._dtype());
            write_scalar_at(b.impl->storage->grad.get(), i, b._dtype(), g_b + g_self);
        }
    }
});
// ------------------- register sub -------------------
register_grad("diff", [](const Tensor& self, const std::vector<Tensor>& parents){
    // parents[0] = a, parents[1] = b
    const Tensor &a = parents[0];
    const Tensor &b = parents[1];
    // self.impl->storage->grad contains dL/dself
    if (!self.impl->storage->grad) throw std::runtime_error("add backward: missing grad for self");

    size_t n = self.numel_();

    // distribute grad to a (respect broadcasting)
    if (a.impl->requires_grad) {
        // allocate if missing (should be ensured by caller, but safe-guard)
        ensure_grad_alloc(const_cast<Tensor&>(a));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double g_a   = read_scalar_at(a.impl->storage->grad.get(), i, a._dtype());
            write_scalar_at(a.impl->storage->grad.get(), i, a._dtype(), g_a - g_self);
        }
    }
    if (b.impl->requires_grad) {
        ensure_grad_alloc(const_cast<Tensor&>(b));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double g_b   = read_scalar_at(b.impl->storage->grad.get(), i, b._dtype());
            write_scalar_at(b.impl->storage->grad.get(), i, b._dtype(), g_b - g_self);
        }
    }
});


// ------------------- register mul -------------------
register_grad("mul", [](const Tensor& self, const std::vector<Tensor>& parents){
    // d(a*b)/da = b * grad_self
    const Tensor &a = parents[0];
    const Tensor &b = parents[1];
    if (!self.impl->storage->grad) throw std::runtime_error("mul backward: missing grad for self");
    size_t n = self.numel_();

    if (a.impl->requires_grad) {
        ensure_grad_alloc(const_cast<Tensor&>(a));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double vb = read_scalar_at(b.impl->storage->data.get(), i, b._dtype());
            double g_a   = read_scalar_at(a.impl->storage->grad.get(), i, a._dtype());
            write_scalar_at(a.impl->storage->grad.get(), i, a._dtype(), g_a + g_self * vb);
        }
    }
    if (b.impl->requires_grad) {
        ensure_grad_alloc(const_cast<Tensor&>(b));
        for (size_t i = 0; i < n; ++i) {
            double g_self = read_scalar_at(self.impl->storage->grad.get(), i, self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            double g_b   = read_scalar_at(b.impl->storage->grad.get(), i, b._dtype());
            write_scalar_at(b.impl->storage->grad.get(), i, b._dtype(), g_b + g_self * va);
        }
    }
});
