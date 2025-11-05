#pragma once
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
inline void ensure_grad_buffer(Tensor &t, bool zero = true);

// Helper: create a tensor whose DATA is copied from self.grad
static Tensor tensor_from_grad(const Tensor& self);
// copy .data -> .grad (allocate grad buffer and copy values)
static void copy_data_to_grad(Tensor &t);

// reduce `t` by summing over axes but keeping dims
static Tensor reduce_sum_axes_keepdims(Tensor t, std::vector<int> axes);

// fetch dimension value of `target` as if left-padded to `nd` dimensions
static size_t dim_in_padded(const Tensor& target, size_t nd, size_t idx);

// accumulate gradient from grad_src into target (broadcast-aware)
void accumulate_grad(Tensor& target, const Tensor& grad_src);

// ------------------ GradFn base ------------------
struct GradFn {
    std::vector<Tensor> parents;              // used for DFS/topo traversal
    virtual void backward(const Tensor& self) = 0; // self is the tensor whose grad is in self.impl->storage->grad
    virtual ~GradFn() = default;
};

// ------------------ Backward nodes ------------------
struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradAdd: missing self grad");
        if (a.requires_grad()) accumulate_grad(a, self);
        if (b.requires_grad()) accumulate_grad(b, self);
    }
};

struct GradSub : GradFn {
    Tensor a, b;
    GradSub(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradMul : GradFn {
    Tensor a, b;
    GradMul(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};


struct GradDiv : GradFn {
    Tensor a, b;
    GradDiv(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradPow : GradFn {
    Tensor a, b;
    GradPow(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradMatMul : GradFn {
    Tensor a, b;
    GradMatMul(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};
struct GradSum : GradFn {
    Tensor t;
    int dim;
    GradSum(const Tensor& t_, int dim_) : t(t_), dim(dim_) { parents = {t}; }

    void backward(const Tensor& self) override ;
};

struct GradLn : GradFn {
    Tensor t;
    GradLn(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradExp : GradFn {
    Tensor t;
    GradExp(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradAddScalar : GradFn {
    Tensor a;
    double scalar;
    GradAddScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradSubScalar : GradFn {
    Tensor a;
    double scalar;
    GradSubScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradSubAfterScalar : GradFn {
    Tensor a;
    double scalar;
    GradSubAfterScalar(double scalar_,  const Tensor& a_) : scalar(scalar_), a(a_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradMulScalar : GradFn {
    Tensor a;
    double scalar;
    GradMulScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradDivScalar : GradFn {
    Tensor a;
    double scalar;
    GradDivScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradDivScalar: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        Tensor grad_input = grad_self * (1.0 / scalar);
        accumulate_grad(a, grad_input);
    }
};
struct GradScalarDiv : GradFn {
    Tensor a;
    double scalar;
    GradScalarDiv(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradScalarDiv: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(), -gv * scalar / (va * va));
        }

        accumulate_grad(a, grad_input);
    }
};
struct GradPowScalar : GradFn {
    Tensor a;
    double scalar;

    GradPowScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradPowScalar: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(),
                            gv * scalar * std::pow(va, scalar - 1.0));
        }

        accumulate_grad(a, grad_input);
    }
};
struct GradScalarPow : GradFn {
    Tensor a;
    double scalar;

    GradScalarPow(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradScalarPow: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(),
                            gv * std::pow(scalar, va) * std::log(scalar));
        }

        accumulate_grad(a, grad_input);
    }
};

// ------------------ topo sort helper ------------------
static void topo_sort_from(const Tensor& root, std::vector<Tensor>& topo);

// ------------------ backward ------------------
void backward(Tensor& loss);
