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

// copy .data -> .grad (allocate grad buffer and copy values)
inline void copy_data_to_grad(Tensor &t);

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

// ------------------ topo sort helper ------------------
void topo_sort_from(const Tensor& root, std::vector<Tensor>& topo);

// ------------------ backward ------------------
void backward(Tensor& loss);
